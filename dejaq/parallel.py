import builtins, datetime, pickle, threading, time, traceback, weakref

import multiprocessing as mp
from multiprocessing import util as _mp_util
import concurrent.futures

import numpy as np
import cloudpickle
from .queues import DejaQueue, NamedSemaphore, ShutDown, Full, _pickle_backend

ctx = mp.get_context("spawn")

def passthrough(item):
    return item

def printer(*args, **kwargs):
    print(args, kwargs)


class WorkerError(RuntimeError):
    """A worker process failed: it died (e.g. killed by the out-of-memory killer), or raised an exception that
    could not be passed on (the original type and message are in the error message)."""


class _RemoteTraceback(Exception):
    """Carries a worker's traceback text; set as the __cause__ of a re-raised worker exception."""

    def __init__(self, tb: str) -> None:
        super().__init__(tb)
        self.tb = tb

    def __str__(self) -> str:
        return self.tb


class _WorkerException:
    """Queue item: an exception raised in a worker (the exception itself, if it can be passed on)."""

    def __init__(self, exc: BaseException) -> None:
        self.tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.type_name = f"{type(exc).__module__}.{type(exc).__qualname__}"
        self.message = str(exc)
        try:
            pickle.loads(_pickle_backend.dumps(exc, protocol=5))  # as the queue will pickle it
            self.exc = exc
        except Exception:
            self.exc = None

    def reraise(self):
        if self.exc is not None:
            raise self.exc from _RemoteTraceback(self.tb)
        raise WorkerError(f"{self.type_name}: {self.message}") from _RemoteTraceback(self.tb)

    def __getstate__(self):
        state = dict(self.__dict__)
        if state["exc"] is not None:
            state["exc"] = state["exc"].with_traceback(None)  # tracebacks cannot be pickled
        return state


class _WorkerDied:
    """Queue item: a worker of this stage exited abnormally (put by the monitor in the creating process)."""

    def __init__(self, pid: int, exitcode: int) -> None:
        self.pid, self.exitcode = pid, exitcode


class _Stop:
    """Queue item: the stop signal at the end of a stage's output (a private type, so that no input item or
    result, e.g. an Ellipsis, can be mistaken for it)."""


class _Zipped:
    """Picklable: zips the given iterables when iterated, in the process that iterates it."""

    def __init__(self, iterables) -> None:
        self._iterables = iterables

    def __iter__(self):
        return builtins.zip(*self._iterables)


def _stop_processes(procs, lock=None, timeout=1.0) -> None:
    """Terminate (if still running) and join processes."""
    with lock or threading.Lock():
        for p in procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        deadline = time.monotonic() + timeout
        for p in procs:
            try:
                p.join(max(0.0, deadline - time.monotonic()))
                if p.is_alive():
                    p.kill()
                    p.join(1.0)
            except Exception:
                pass


# One monitor thread per process watches the workers of the lazymaps this process created (only the parent of
# a process sees its exit code). When a worker exits abnormally, the monitor tells the stage's consumer by
# putting a _WorkerDied item into the stage's output queue. A worker that finished normally has marked itself as
# done (after putting its stop signal into the queue); any other exit, even with exit code 0, counts as a death. If the consumer is a worker of a downstream
# stage, it passes the error on as an item, so it reaches the end of the pipeline.
_MONITOR_INTERVAL = 0.5
_monitored: "weakref.WeakSet" = weakref.WeakSet()
_monitor_lock = threading.Lock()
_monitor_thread = None


def _check_all() -> None:
    # A function of its own, so that no reference to a lazymap outlives the check (it would keep it alive).
    for lzy in list(_monitored):
        try:
            lzy._check_workers()
        except Exception:  # the monitor must not die
            pass


def _monitor() -> None:
    while True:
        time.sleep(_MONITOR_INTERVAL)
        _check_all()


def _watch(lzy) -> None:
    global _monitor_thread
    with _monitor_lock:
        _monitored.add(lzy)
        if _monitor_thread is None or not _monitor_thread.is_alive():
            _monitor_thread = threading.Thread(target=_monitor, name="dejaq-worker-monitor", daemon=True)
            _monitor_thread.start()


class lazymap:
    ''' Returns an Iterable, functionally related to map (except that outputs are calculated by a pool of processes).

    Results are returned in input order. If a worker raises an exception, iterating raises it (with the worker's
    traceback as its cause); if a worker dies, iterating raises WorkerError. In both cases the stage's workers
    are stopped. They are also stopped when the lazymap is closed (close(), or the end of a ``with`` block),
    garbage-collected, or at interpreter exit. Leaving a loop early does not stop them: iterating again
    continues with the next result. Iterating a closed lazymap raises RuntimeError.

    Args:
        fcn (callable): function that is being mapped. Signature: fcn(item, **kwargs)
        it (iterable): iterable that maps over the function, providing items as arguments (must be picklable)
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
        start_type (str): when the workers start reading input: 'lazy' (default) when iteration starts,
            'eager' at once, 'manual' when start() is called

        **kwargs: optional, being passed to fcn

    Returns:
        (iterable): an iterable that returns the results of fcn(item) for each item in it
    '''

    def __init__(self, fcn, it, n_workers=1, buffer_bytes=10e6, start_type='lazy', **kwargs):
        if not isinstance(n_workers, int) or n_workers < 1:
            raise ValueError(f"n_workers must be a positive integer, not {n_workers!r}")
        if start_type not in ("lazy", "eager", "manual"):
            raise ValueError(f"start_type must be 'lazy', 'eager' or 'manual', not {start_type!r}")
        self._own_input = None
        self._closed = False
        if n_workers > 1 and not isinstance(it, lazymap):
            # one process reads the input, so that a stateful iterable (e.g. a file reader) is read only once
            it = self._own_input = lazymap(None, it, n_workers=1, buffer_bytes=buffer_bytes)
        self._it = it
        self._start_type = start_type
        self._out_queue = DejaQueue(buffer_bytes)
        self._in_sem = [NamedSemaphore() for _ in range(n_workers)]
        self._out_sem = [NamedSemaphore() for _ in range(n_workers)]
        if start_type == "eager":
            self._in_sem[0].release()
            self._started = True
            self._start_gate = None
        else:
            self._started = False
            # Pickled copies of this lazymap (one in each worker of a downstream stage) all try to start it; the
            # gate lets only the first one through, so that exactly one input token circulates.
            self._start_gate = NamedSemaphore(initial=1)
        self._out_sem[0].release()
        self._n_workers = n_workers
        self._done = ctx.RawArray("b", n_workers)  # set by each worker when it finished normally
        pkl = cloudpickle.dumps((fcn, self._it))
        worker_args = [pkl, self._out_queue, n_workers, self._in_sem, self._out_sem, self._done]
        procs = [ctx.Process(target=self._worker_fcn, args=(pid, *worker_args), kwargs=kwargs) for pid in range(n_workers)]
        for p in procs:
            p.start()
        self.pids = [p.pid for p in procs]
        self._procs = procs
        self._owner = True  # only the creating process stops the workers (not pickled copies in other workers)
        self._closing = False
        self._reported = set()
        self._procs_lock = threading.Lock()  # exit codes are read by the monitor, workers joined by close()
        # Stops the workers when this object is garbage-collected, or at interpreter exit before Python waits for
        # its child processes (which would otherwise wait forever for workers blocked on a full queue).
        self._finalizer = _mp_util.Finalize(self, _stop_processes, args=(procs, self._procs_lock), exitpriority=10)
        _watch(self)

    def __len__(self):
        return len(self._it)

    def __iter__(self):
        """ Returns an iterator over the results of fcn(item) for each item in it """
        if self._closed:
            raise RuntimeError("this lazymap was closed (its workers were stopped)")
        if self._start_type == "lazy":
            self._start()
        return self._results()

    def _start(self):
        if not self._started:
            if self._start_gate is None or self._start_gate.acquire(timeout=0):
                self._in_sem[0].release()
            self._started = True

    def _results(self):
        try:
            while True:
                try:
                    item = self._out_queue.get()
                except ShutDown:  # close() shuts the queue down, also for consumers in other threads or processes
                    raise RuntimeError("this lazymap was closed (its workers were stopped)") from None
                kind = type(item)
                if kind is _Stop:  # put it back for other consumers of this queue
                    try:
                        self._out_queue.put(item)
                    except ShutDown:
                        pass
                    return
                if kind is _WorkerException:
                    item.reraise()
                if kind is _WorkerDied:
                    raise WorkerError(f"worker process {item.pid} died (exit code {item.exitcode})")
                yield item
        except GeneratorExit:  # the loop was left early: the workers keep going, iterating again continues
            raise
        except BaseException:
            if getattr(self, "_owner", False):
                self.close()  # a worker failed (or e.g. Ctrl-C while waiting): stop the workers
            raise

    @staticmethod
    def _worker_fcn(pid, pkl, _out_queue, _n_workers, in_sem, out_sem, done, **kwargs):
        # A failure is passed on as a _WorkerException item, in this worker's turn, after which the worker stops.
        failure = None
        try:
            fcn, it = cloudpickle.loads(pkl)
        except BaseException as e:
            failure = _WorkerException(e)
        nxt = (pid + 1) % _n_workers
        in_sem[pid].acquire()
        if failure is None:
            try:
                iterator = iter(it)
            except BaseException as e:
                failure = _WorkerException(e)
        in_sem[pid].release()

        while True:
            in_sem[pid].acquire()
            if failure is not None:
                in_sem[nxt].release()
                res = failure
            else:
                try:
                    item = next(iterator)
                except StopIteration:
                    in_sem[nxt].release()
                    break
                except BaseException as e:  # the input failed (e.g. an upstream stage's worker raised or died)
                    in_sem[nxt].release()
                    res = failure = _WorkerException(e)
                else:
                    in_sem[nxt].release()
                    try:
                        res = item if fcn is None else fcn(item, **kwargs)
                    except Exception as e:
                        res = _WorkerException(e)
                    except BaseException as e:  # e.g. sys.exit() in fcn: pass it on, then stop
                        res = failure = _WorkerException(e)
            out_sem[pid].acquire()
            try:
                _out_queue.put(res)
            except Exception as e:  # e.g. the result cannot be pickled
                _out_queue.put(_WorkerException(e))
            out_sem[nxt].release()
            if failure is not None:
                break

        out_sem[pid].acquire()
        try:
            _out_queue.put(_Stop())
        except ShutDown:
            pass
        out_sem[nxt].release()
        done[pid] = 1

    def _check_workers(self) -> None:
        """Called by the monitor thread: report workers that exited abnormally to the consumer."""
        if self._closing:
            return
        with self._procs_lock:
            for i, p in enumerate(self._procs):
                if p.pid in self._reported:
                    continue
                code = p.exitcode
                if code is None or (code == 0 and self._done[i]):
                    continue
                try:
                    self._out_queue.put(_WorkerDied(p.pid, code), timeout=0.2)
                except (Full, ShutDown):
                    continue  # try again at the next check
                self._reported.add(p.pid)

    def close(self):
        ''' Stops the workers (if still running) and waits for them to end. Also closes the input stage that was
        created for n_workers > 1. Can be called repeatedly; also called at the end of a ``with`` block.
        '''
        if not getattr(self, "_owner", False):
            return
        self._closing = self._closed = True
        _monitored.discard(self)
        self._finalizer()
        try:
            self._out_queue.shutdown()  # wakes consumers waiting for results (they raise RuntimeError)
        except Exception:
            pass
        if self._own_input is not None:
            self._own_input.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        for k in ("_procs", "_finalizer", "_procs_lock", "_reported", "_done"):
            state.pop(k, None)
        state["_owner"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def compute(self, progress=True, ndarray=False, tqdm_args=None):
        ''' Computes the results of the lazymap and returns them as a list (or a numpy array)

        Args:
            progress (bool): whether to show a tqdm progress bar (default: True)
            ndarray (bool): whether to try to return the results as a numpy array (default: False)
            tqdm_args (dict): keyword arguments for tqdm

        Returns:
            (list): a list of the results of fcn(item) for each item in it
        '''
        iterable = self
        if progress:
            from tqdm.auto import tqdm
            iterable = tqdm(iterable, **(tqdm_args or {}))
        out = list(iterable)
        if ndarray:
            try:
                return np.array(out)
            except Exception:
                return out
        return out

    def run(self, progress=True, ndarray=False, tqdm_args=None):
        ''' Same as compute(): runs the lazymap and returns its results as a list (or a numpy array). '''
        return self.compute(progress=progress, ndarray=ndarray, tqdm_args=tqdm_args)

    def submit(self):
        ''' Submits the computation to a background thread and returns a Future object.

        Returns:
            (Future): a Future object that can be used to fetch the result later
        '''
        future = concurrent.futures.Future()
        def run():
            try:
                result = self.compute(progress=False)
                future.set_result(result)
            except BaseException as e:
                future.set_exception(e)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return future

    def map(self, fcn, n_workers=1, buffer_bytes=10e6, **kwargs):
        ''' Maps another function over the output of this lazymap, returning a new lazymap.

        Args:
            fcn (callable): function that is being mapped. Signature: fcn(item, **kwargs)
            n_workers (int): number of workers (default: 1)
            buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
            **kwargs: optional, being passed to fcn

        Returns:
            (lazymap): a new lazymap that maps fcn over the output of this lazymap
        '''
        return lazymap(fcn, self, n_workers, buffer_bytes, **kwargs)

    def start(self):
        ''' Starts the lazymap if it was created with start_type='manual'.
        '''
        self._start()
        return self

    def tee(self, n):
        """A helper to tee this lazymap iterable into multiple independent lazymap iterables.
        Args:
            n (int): number of independent lazymap iterables to create
        """
        return tee(self, n)


def Parallel(n_workers=1, buffer_bytes=10e6):
    ''' A wrapper to make a class or callablle a parallel worker. Can be used as a decorator.

    Args:
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)

    Returns:

    '''
    def decorator(cls):
        if isinstance(cls, type):
            return WorkerWrapper(cls, n_workers, buffer_bytes)
        elif callable(cls):
            def wrapped(iterable, **kwargs):
                return lazymap(cls, iterable, n_workers, buffer_bytes, **kwargs)
            return wrapped
        else:
            raise ValueError(f'Invalid type {type(cls)}')
    return decorator

class WorkerWrapper:
    ''' A helper class used by the Parallel decorator to wrap a class to make it a parallel worker.

    Args:
        cls (class): the class to be wrapped
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
    '''
    def __init__(self, cls, n_workers=1, buffer_bytes=10e6):
        self._instance = None
        self._cls = cls
        self._n_workers = n_workers
        self._buffer_bytes = int(buffer_bytes)
    def __call__(self, *init_args, **init_kwargs):
        def mapper(iterable, **map_kwargs):
            def worker(arg, **kwargs):
                if self._instance is None:
                    self._instance = self._cls(*init_args, **init_kwargs)
                return self._instance(arg, **kwargs)
            return lazymap(worker, iterable, self._n_workers, self._buffer_bytes, **map_kwargs)
        return mapper


class Startable:
    """ Wrap an iterable so iteration blocks until .start() is called. Works across processes.

    Args:
        source (iterable or callable): an iterable or a zero-arg factory function that produces
        emit_one (bool): if True, iteration will pause after each item until .start() is called again
        ctx (str): multiprocessing context (default: "spawn")
    """
    def __init__(self, source, emit_one=False):
        self._start = ctx.Event()
        self._done  = ctx.Event()
        self._source = source      # iterable OR zero-arg factory
        self._it = None
        self._emit_one = ctx.Value("b", emit_one)

    def start(self):
        self._start.set()
        return self

    def pause(self):
        self._start.clear()
        return self

    def stop(self):
        self._done.set()
        self._start.set()          # release any waiters
        return self

    def __iter__(self):
        return self

    def __next__(self):
        self._start.wait()
        if self._done.is_set():
            raise StopIteration
        if self._it is None:
            src = self._source() if callable(self._source) else self._source
            self._it = iter(src)   # created in the consumer process
        if self._emit_one.value:
            self._start.clear()
        return next(self._it)

class Ticker(Startable):
    """ An iterable that yields the current datetime when started.
    """
    def __init__(self, emit_one=False):
        super().__init__(iter(datetime.datetime.now, object()), emit_one=emit_one)

def _tee_worker(iterable, queues):
        for item in iterable:
            for q in queues:
                q.put(item)
        for q in queues:
            q._signal_stop()

def tee(lzy, n):
    """A helper to tee a lazymap iterable into multiple independent lazymap iterables.
    Args:
        lzy (lazymap): the lazymap iterable to tee
        n (int): number of independent lazymap iterables to create
    """

    class QueueIterator:
        """A picklable wrapper that iterates over a DejaQueue."""

        def __init__(self, queue):
            self._queue = queue

        def __iter__(self):
            for item in self._queue:
                yield item
            self._queue._signal_stop()

    queues = [DejaQueue() for _ in range(n)]
    p = ctx.Process(target=_tee_worker, args=(lzy, queues))
    p.start()
    _mp_util.Finalize(None, _stop_processes, args=([p],), exitpriority=10)  # at interpreter exit, at the latest
    return [lazymap(None, QueueIterator(q), n_workers=1) for q in queues]

def zip(*lazys):
    """Zips several lazymap iterables into one lazymap of tuples, in a process of its own: the items do not pass
    through the calling process, and the result can be fed into further stages (unlike a builtin zip object,
    which holds generators that cannot be passed to worker processes). To consume several pipelines in the
    calling process, the builtin zip is enough.

    Args:
        *lazys (lazymap): the lazymap iterables to zip
    """
    return lazymap(None, _Zipped(lazys), n_workers=1)
