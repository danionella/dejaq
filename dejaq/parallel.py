import builtins, datetime, os, pickle, threading, time, traceback, weakref

import multiprocessing as mp
from multiprocessing import util as _mp_util
import concurrent.futures

import numpy as np
import cloudpickle
import psutil
from .queues import DejaQueue, NamedSemaphore, ShutDown, Full, _pickle_backend

ctx = mp.get_context("spawn")

class WorkerError(RuntimeError):
    """A worker process failed: it died (e.g. killed by the out-of-memory killer), or raised an exception that
    could not be passed on (the original type and message are in the error message)."""


class _RemoteTraceback(Exception):
    """The worker's traceback text (the __cause__ of a re-raised worker exception)."""

    def __str__(self):
        return self.args[0]


class _WorkerException:
    """Queue item: an exception raised in a worker (as WorkerError if it cannot be passed on)."""

    def __init__(self, exc: BaseException, fallback=None) -> None:
        self.tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            pickle.loads(_pickle_backend.dumps(exc, protocol=5))  # as the queue will pickle it
            self.exc = exc
        except Exception:
            self.exc = (fallback or WorkerError)(f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}")

    def reraise(self):
        raise self.exc from _RemoteTraceback(self.tb)


class _Stop:
    """Queue item: the stop signal at the end of a stage's output (a private type, so that no input item or
    result, e.g. an Ellipsis, can be mistaken for it)."""


class _Zipped:
    """Picklable: zips the given iterables when iterated, in the process that iterates it."""

    def __init__(self, iterables) -> None:
        self._iterables = iterables

    def __iter__(self):
        return builtins.zip(*self._iterables)


def _close_input(it) -> None:
    """Close the stages that feed a stage (each stage has a single consumer, so they are of no further use)."""
    if isinstance(it, lazymap):
        it.close()  # a no-op for pickled copies: only the creating process closes
    elif isinstance(it, _Zipped):
        for x in it._iterables:
            _close_input(x)


_PARENT_CHECK_INTERVAL = 0.5
_STOP_GRACE = 0.1  # seconds that close() gives workers to end by themselves after their queue was shut down
_EXIT_GRACE = 10.0  # ... and workers that finished their input (e.g. a class worker that finalizes a file in __del__)


def _exit_with_parent(ppid) -> None:
    """In a worker: exit once the creating process is gone (it crashed, or terminated this worker's parent
    stage), so that no worker stays blocked on a queue (POSIX: the orphan is adopted; Windows: not running)."""
    try:
        proc = psutil.Process(ppid)
    except psutil.Error:
        os._exit(1)

    parent = os.getppid()  # (forkserver: not the creating process)

    def watch():
        while os.getppid() == parent and proc.is_running():
            time.sleep(_PARENT_CHECK_INTERVAL)
        os._exit(1)

    threading.Thread(target=watch, name="dejaq-parent-watchdog", daemon=True).start()


def _forwardable(it) -> bool:
    """Whether a stage's input is read from queues (one stage, or zipped stages), so that _forward can be used."""
    if isinstance(it, _Zipped):
        return all(isinstance(x, lazymap) for x in it._iterables)
    return isinstance(it, lazymap)


def _forward(it, outs, fcn=None, kwargs=None) -> None:
    """Pass on the results of a stage (or tuples of the results of zipped stages), or fcn of them, to each queue
    in outs without a private copy: the item (and the result) is handled inside get(callback=...), while its
    arrays still view the source queue's memory, so they are copied once, straight from queue to queue. Ends
    like zip(): at the end of any source; then puts the stop signal (an error before it)."""
    kwargs = kwargs or {}
    sources = it._iterables if isinstance(it, _Zipped) else [it]
    live = list(outs)

    def put(res):
        for q in list(live):
            try:
                q.put(res)  # (before the source's memory is released: also a view of the item is safe)
            except ShutDown:  # e.g. a tee branch was closed
                live.remove(q)
            except Exception as e:  # e.g. the result cannot be pickled
                q.put(_WorkerException(e))

    def take(i, items):
        def got(item):
            kind = type(item)
            if kind is _Stop:
                return False
            if kind is _WorkerException:
                put(item)  # the consumer raises it
                return False
            items.append(item)
            if i + 1 < len(sources):
                return take(i + 1, items)
            res = tuple(items) if isinstance(it, _Zipped) else items[0]
            go_on = True
            if fcn is not None:
                try:
                    res = fcn(res, **kwargs)
                except Exception as e:
                    res = _WorkerException(e)
                except BaseException as e:  # e.g. sys.exit() in fcn: pass it on, then stop
                    res, go_on = _WorkerException(e), False
            put(res)
            return go_on and bool(live)
        try:
            return sources[i]._out_queue.get(callback=got)
        except ShutDown:
            raise RuntimeError("a stage feeding this one was closed") from None

    try:
        for src in sources:
            if src._start_type == "lazy":  # (as iterating it would)
                src._start()
        while take(0, []):
            pass
    except BaseException as e:
        put(_WorkerException(e))
    for q in live:
        _put_stop(q)


def _keep(obj):
    return obj


def _drop(obj):
    return None


def _put_stop(q) -> None:
    try:
        q.put(_Stop())
    except ShutDown:
        pass


def _stop_processes(procs, lock=None, timeout=1.0, grace=0.0, done=None) -> None:
    """Wait up to `grace` s for the processes to end (those with done[i] set: up to _EXIT_GRACE s), terminate those
    left, and kill those alive after `timeout` s."""
    with lock or threading.Lock():
        for wait, stop in ((grace, "terminate"), (timeout, "kill"), (1.0, None)):
            end = time.monotonic() + wait
            for p in procs:
                p.join(max(0.0, end - time.monotonic()))
            if done and stop == "terminate":  # (read after the grace: a worker sets done just after its end marker)
                end = time.monotonic() + _EXIT_GRACE
                for p, d in builtins.zip(procs, done):
                    if d:
                        p.join(max(0.0, end - time.monotonic()))
            for p in procs:
                if stop and p.is_alive():
                    getattr(p, stop)()


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
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes); for large items (e.g. camera frames) about
            10 times the item size or more
        start_type (str): when the workers start reading input: 'lazy' (default) when iteration starts,
            'eager' at once, 'manual' when start() is called
        ordered (bool): if True (default), results are returned in input order; if False, in the order in which
            they are ready (a slow item does not hold up the others)
        zero_copy (bool): n_workers=1 only, with a stage (or a zip of stages) as input: fcn receives the item without
            a copy, its numpy arrays viewing the input queue's memory. They are valid only during the call: do not
            keep them (e.g. as the previous frame) or hand them to another thread; copy what you need to keep.
            Returning them (or views of them) is fine. Default False.

        **kwargs: optional, being passed to fcn

    Returns:
        (iterable): an iterable that returns the results of fcn(item) for each item in it
    '''

    def __init__(self, fcn, it, n_workers=1, buffer_bytes=10e6, start_type='lazy', ordered=True, zero_copy=False,
                 **kwargs):
        if not isinstance(n_workers, int) or n_workers < 1:
            raise ValueError(f"n_workers must be a positive integer, not {n_workers!r}")
        if zero_copy and n_workers != 1:
            raise ValueError("zero_copy=True needs n_workers=1 (the function runs while the input item is held)")
        if start_type not in ("lazy", "eager", "manual"):
            raise ValueError(f"start_type must be 'lazy', 'eager' or 'manual', not {start_type!r}")
        self._closed = False
        if n_workers > 1 and not isinstance(it, lazymap):
            # one process reads the input, so that a stateful iterable (e.g. a file reader) is read only once
            it = lazymap(None, it, n_workers=1, buffer_bytes=buffer_bytes)
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
        self._buffer_bytes = buffer_bytes
        self._done = ctx.RawArray("b", n_workers)  # set by each worker when it finished normally
        pkl = cloudpickle.dumps((fcn, self._it, kwargs))
        # ordered=False: workers share their input queue (whoever is free takes the next item); the last one to
        # finish puts the stop signal (kept referenced here until the workers have opened it)
        self._finished = finished = None if ordered else NamedSemaphore(initial=n_workers - 1)
        worker_args = (pkl, self._out_queue, n_workers, self._in_sem, self._out_sem, self._done, ordered, finished,
                       zero_copy)
        self._procs = procs = []
        self._owner = True  # only the creating process stops the workers (not pickled copies in other workers)
        self._reported = set()
        self._procs_lock = threading.Lock()  # exit codes are read by the monitor, workers joined by close()
        # stops the workers at close(), garbage collection (also if starting them fails part-way) or interpreter
        # exit, before Python waits for its child processes (which may be blocked on a full queue)
        self._finalizer = _mp_util.Finalize(self, _stop_processes, args=(procs, self._procs_lock),
                                            kwargs={"grace": _STOP_GRACE, "done": self._done}, exitpriority=10)
        for pid in range(n_workers):
            p = ctx.Process(target=self._worker_fcn, args=(os.getpid(), pid, *worker_args))
            p.start()
            procs.append(p)
        self.pids = [p.pid for p in procs]
        _watch(self)

    def __len__(self):
        return len(self._it)

    def qsize(self) -> int:
        """Number of results waiting in this stage's output queue (e.g. to see whether it keeps up; it may be
        one too high while the stage finishes)."""
        # without the end markers: one per worker of an ordered stage (or tee branch), one for an unordered stage
        markers = sum(self._done) if getattr(self, "_finished", None) is None else all(self._done)
        return max(0, self._out_queue.qsize() - markers)

    def __iter__(self):
        """ Returns an iterator over the results of fcn(item) for each item in it """
        if self._closed:
            raise RuntimeError("this lazymap was closed (its workers were stopped)")
        if self._start_type == "lazy":
            self._start()
        return self._results()

    def iter(self, zero_copy=False):
        """ Returns an iterator over the results, like iter(stage).

        Args:
            zero_copy (bool): if True, each result's numpy arrays view the stage's queue memory instead of being
                copied out. They are valid until the next result is requested: do not keep them (copy what you need
                to keep). Default False.
        """
        if not zero_copy:
            return iter(self)
        if self._closed:
            raise RuntimeError("this lazymap was closed (its workers were stopped)")
        if self._start_type == "lazy":
            self._start()
        return self._views()

    def _views(self):
        q = self._out_queue
        held = False  # the item handed out last is still in the queue (its memory stays valid until removed)
        try:
            while True:
                if held:
                    q.get(callback=_drop)
                    held = False
                try:
                    item = q.get(peek_only=True, callback=_keep)
                except ShutDown:
                    raise RuntimeError("this lazymap was closed (its workers were stopped)") from None
                kind = type(item)
                if kind is _Stop:
                    return  # (left in the queue for other consumers)
                if kind is _WorkerException:
                    q.get(callback=_drop)
                    item.reraise()
                held = True
                yield item
        except GeneratorExit:  # left early: the item handed out last counts as taken
            if held:
                q.get(callback=_drop)
            raise
        except BaseException:
            if getattr(self, "_owner", False):
                self.close()
            raise

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
                yield item
        except GeneratorExit:  # the loop was left early: the workers keep going, iterating again continues
            raise
        except BaseException:
            if getattr(self, "_owner", False):
                self.close()  # a worker failed (or e.g. Ctrl-C while waiting): stop the workers
            raise

    @staticmethod
    def _worker_fcn(parent, *args):
        _exit_with_parent(parent)
        try:
            lazymap._worker_body(*args)
        except ShutDown:  # the stage was closed
            pass

    @staticmethod
    def _worker_body(pid, pkl, _out_queue, _n_workers, in_sem, out_sem, done, ordered, finished, zero_copy):
        # Unless ordered=False, workers take turns (round-robin, passing semaphore tokens) to read the input and
        # to put results. A failure is passed on as a _WorkerException item, in this worker's turn, after which
        # the worker stops.
        read_in_turn = write_in_turn = ordered
        failure = None
        try:
            fcn, it, kwargs = cloudpickle.loads(pkl)
        except BaseException as e:
            failure = _WorkerException(e)
        nxt = (pid + 1) % _n_workers
        in_sem[pid].acquire()  # the start signal
        if failure is None and _n_workers == 1 and (fcn is None or zero_copy) and _forwardable(it):  # e.g. zip()
            _forward(it, [_out_queue], fcn, kwargs)
            done[pid] = 1
            return
        if failure is None:
            try:
                iterator = iter(it)
            except BaseException as e:
                failure = _WorkerException(e)
        if read_in_turn:
            in_sem[pid].release()
        else:
            in_sem[nxt].release()  # pass the start signal on

        while True:
            if read_in_turn:
                in_sem[pid].acquire()
            if failure is not None:
                res = failure
            else:
                try:
                    item = next(iterator)
                except StopIteration:
                    if read_in_turn:
                        in_sem[nxt].release()
                    break
                except BaseException as e:  # the input failed (e.g. an upstream stage's worker raised or died)
                    res = failure = _WorkerException(e)
            if read_in_turn:
                in_sem[nxt].release()
            if failure is None:
                try:
                    res = item if fcn is None else fcn(item, **kwargs)
                except Exception as e:
                    res = _WorkerException(e)
                except BaseException as e:  # e.g. sys.exit() in fcn: pass it on, then stop
                    res = failure = _WorkerException(e)
            if write_in_turn:
                out_sem[pid].acquire()
            try:
                _out_queue.put(res)
            except Exception as e:  # e.g. the result cannot be pickled
                _out_queue.put(_WorkerException(e))
            if write_in_turn:
                out_sem[nxt].release()
            if failure is not None:
                break

        if write_in_turn:
            out_sem[pid].acquire()
            _put_stop(_out_queue)
            out_sem[nxt].release()
        elif not finished.acquire(timeout=0):  # the last worker to finish (it holds n_workers - 1 counts)
            _put_stop(_out_queue)
        done[pid] = 1

    def _check_workers(self) -> None:
        """Called by the monitor thread: report workers that exited abnormally to the consumer."""
        if self._closed:
            return
        with self._procs_lock:
            for i, p in enumerate(self._procs):
                if p.pid in self._reported:
                    continue
                code = p.exitcode
                if code is None or (code == 0 and self._done[i]):
                    continue
                try:
                    died = WorkerError(f"worker process {p.pid} died (exit code {code})")
                    self._out_queue.put(_WorkerException(died), timeout=0.2)
                except (Full, ShutDown):
                    continue  # try again at the next check
                self._reported.add(p.pid)

    def close(self):
        ''' Stops the workers (if still running) and waits for them to end. Also closes the stages that feed this
        one (created in this process). Can be called repeatedly; also called at the end of a ``with`` block.
        '''
        if not getattr(self, "_owner", False) or self._closed:
            return
        self._closed = True
        _monitored.discard(self)
        # Order matters: a process that is terminated inside a queue operation can leave that queue's internal
        # lock held, so a queue is shut down while every process that uses it still runs. Hence the input stages
        # are closed first (this stage's workers read from them), and then this stage's queue is shut down
        # (waking workers blocked on it, which then end, and consumers in other threads, which raise
        # RuntimeError) before the workers are stopped.
        _close_input(self._it)
        try:
            self._out_queue.shutdown()
        except Exception:
            pass
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        for k in ("_procs", "_finalizer", "_procs_lock", "_reported", "_done", "_tee", "_it"):
            state.pop(k, None)
        state["_owner"] = False
        return state

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

    run = compute

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

    def map(self, fcn, n_workers=1, buffer_bytes=10e6, ordered=True, zero_copy=False, **kwargs):
        ''' Maps another function over the output of this lazymap, returning a new lazymap.

        Args:
            fcn (callable): function that is being mapped. Signature: fcn(item, **kwargs)
            n_workers (int): number of workers (default: 1)
            buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
            ordered (bool): results in input order (default); False: as soon as they are ready
            zero_copy (bool): with n_workers=1, fcn gets each item as a view of the input queue (see lazymap)
            **kwargs: optional, being passed to fcn

        Returns:
            (lazymap): a new lazymap that maps fcn over the output of this lazymap
        '''
        return lazymap(fcn, self, n_workers, buffer_bytes, ordered=ordered, zero_copy=zero_copy, **kwargs)

    def start(self):
        ''' Starts the lazymap if it was created with start_type='manual'.
        '''
        self._start()
        return self

    def tee(self, n, buffer_bytes=None):
        """Splits this lazymap into n lazymaps that each yield all of its results (see dejaq.parallel.tee)."""
        return tee(self, n, buffer_bytes)


def Parallel(n_workers=1, buffer_bytes=10e6, ordered=True, zero_copy=False):
    ''' A wrapper to make a class or callable a parallel worker. Can be used as a decorator.

    Args:
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
        ordered (bool): if True (default), results come in input order; if False, as soon as they are ready
        zero_copy (bool): n_workers=1 only: the function receives its input without a copy, valid only during the
            call (see lazymap)

    Returns:

    '''
    def decorator(cls):
        if isinstance(cls, type):
            return WorkerWrapper(cls, n_workers, buffer_bytes, ordered, zero_copy)
        elif callable(cls):
            def wrapped(iterable, **kwargs):
                return lazymap(cls, iterable, n_workers, buffer_bytes, ordered=ordered, zero_copy=zero_copy, **kwargs)
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
    def __init__(self, cls, n_workers=1, buffer_bytes=10e6, ordered=True, zero_copy=False):
        self._instance = None
        self._cls = cls
        self._n_workers = n_workers
        self._buffer_bytes = int(buffer_bytes)
        self._ordered = ordered
        self._zero_copy = zero_copy
    def __call__(self, *init_args, **init_kwargs):
        def mapper(iterable, **map_kwargs):
            def worker(arg, **kwargs):
                if self._instance is None:
                    self._instance = self._cls(*init_args, **init_kwargs)
                return self._instance(arg, **kwargs)
            return lazymap(worker, iterable, self._n_workers, self._buffer_bytes, ordered=self._ordered,
                           zero_copy=self._zero_copy, **map_kwargs)
        return mapper


def _ticks():
    return iter(datetime.datetime.now, None)  # now() never returns None: endless


class Startable:
    """ Wraps an iterable so that iteration waits until start() is called. Works across processes: it can be
    the input of a stage (iterated in a worker) and be started, paused and stopped from the main process.

    Args:
        source (iterable or callable): an iterable, or a zero-argument function that returns one (called where
            the Startable is first iterated, e.g. in a worker)
        emit_one (bool): if False (default), items are emitted from start() until pause() or stop(); if True, each
            start() lets one more item through

    Call start(), pause() and stop() from one process (e.g. the one that created the Startable).
    """
    def __init__(self, source, emit_one=False):
        self._source = source  # iterable OR zero-arg factory
        self._emit_one = bool(emit_one)
        # continuous: the gate holds 1 count while running (each item passes it); emit_one: 1 count per item
        self._gate = NamedSemaphore()
        self._stopped = NamedSemaphore()  # holds 1 count once stopped
        self._running = False  # (known in the controlling process)
        self._it = None

    def start(self):
        if self._emit_one or not self._running:
            self._gate.release()
            self._running = not self._emit_one
        return self

    def pause(self):
        if self._emit_one:
            while self._gate.acquire(timeout=0):  # withdraw items not yet emitted
                pass
        elif self._running:
            self._gate.acquire()
            self._running = False
        return self

    def stop(self):
        self._stopped.release()
        self._gate.release()  # wake a waiting consumer
        return self

    def __iter__(self):
        return self

    def __next__(self):
        self._gate.acquire()
        if self._stopped.acquire(timeout=0):
            self._stopped.release()  # stays stopped
            self._gate.release()  # for the next consumer (or next call)
            raise StopIteration
        if not self._emit_one:
            self._gate.release()
        if self._it is None:
            src = self._source() if callable(self._source) else self._source
            self._it = iter(src)  # created in the consumer process
        return next(self._it)


class Ticker(Startable):
    """ An iterable that yields the current datetime while started (see Startable).
    """
    def __init__(self, emit_one=False):
        super().__init__(_ticks, emit_one=emit_one)

def _tee_worker(parent, pkl, queues, done):
    _exit_with_parent(parent)
    _forward(cloudpickle.loads(pkl), queues)  # each branch raises errors and deaths of the upstream stage
    done[0] = 1


class _Tee:
    """Shared by the branches of a tee: the upstream stage, and the process that copies its results into the
    branches' queues. Stopped when all branches are closed or garbage-collected."""

    def __init__(self, lzy, n, buffer_bytes):
        self.upstream = lzy  # keeps the upstream stage alive as long as a branch is
        self.buffer_bytes = buffer_bytes
        self.queues = [DejaQueue(buffer_bytes) for _ in range(n)]
        self.done = ctx.RawArray("b", 1)
        self.proc = ctx.Process(target=_tee_worker, args=(os.getpid(), cloudpickle.dumps(lzy), self.queues, self.done))
        self.proc.start()
        self.open_branches = set(range(n))
        self.lock = threading.Lock()
        self.finalizer = _mp_util.Finalize(self, _stop_processes, args=([self.proc],), exitpriority=10)

    def branch_closed(self, i):
        self.queues[i].shutdown()  # the tee process drops this branch (also if it was not closed, but collected)
        with self.lock:
            self.open_branches.discard(i)
            last = not self.open_branches
        if last:  # (in this order: see lazymap.close)
            _close_input(self.upstream)
            self.finalizer.cancel()
            _stop_processes([self.proc], grace=_STOP_GRACE)


class _TeeBranch(lazymap):
    """One output of tee(): a lazymap whose results the tee process puts into its queue."""

    def __init__(self, tee_, i):
        self._tee = tee_
        self._closed = False
        self._it = None  # (closing a branch does not close the upstream stage: see _Tee.branch_closed)
        self._start_type, self._started, self._start_gate = "eager", True, None
        self._out_queue = tee_.queues[i]
        self._buffer_bytes = tee_.buffer_bytes
        self._done = tee_.done
        self._procs = [tee_.proc]
        self.pids = [tee_.proc.pid]
        self._owner = True
        self._reported = set()
        self._procs_lock = threading.Lock()
        self._finalizer = _mp_util.Finalize(self, tee_.branch_closed, args=(i,), exitpriority=11)
        _watch(self)

    def __len__(self):
        return len(self._tee.upstream)


def tee(lzy, n, buffer_bytes=None):
    """Splits a lazymap into n lazymaps that each yield all of its results, like itertools.tee.

    A process of its own copies each result into the n branches' queues. As with itertools.tee, consume the
    branches side by side: if one branch falls behind by more than its queue holds, the others wait for it.

    Args:
        lzy (lazymap): the lazymap to split
        n (int): number of branches
        buffer_bytes (int): queue size of each branch (default: that of lzy)
    """
    if buffer_bytes is None:
        buffer_bytes = getattr(lzy, "_buffer_bytes", None) or 10e6
    t = _Tee(lzy, n, buffer_bytes)
    return [_TeeBranch(t, i) for i in range(n)]


def zip(*lazys, buffer_bytes=None):
    """Zips several lazymap iterables into one lazymap of tuples, in a process of its own: the items do not pass
    through the calling process, and the result can be fed into further stages (unlike a builtin zip object,
    which holds generators that cannot be passed to worker processes). To consume several pipelines in the
    calling process, the builtin zip is enough. Each item is copied once (see the README for measurements).

    Args:
        *lazys (lazymap): the lazymap iterables to zip (each one once: to use a stage twice, tee() it)
        buffer_bytes (int): size of the zipped stage's queue (default: the sum of the inputs' buffers)
    """
    if len({id(x) for x in lazys}) < len(lazys):
        raise ValueError("zip() got the same stage twice: use stage.tee() to read it twice")
    buffer_bytes = buffer_bytes or sum(getattr(x, "_buffer_bytes", 10e6) for x in lazys)
    return lazymap(None, _Zipped(lazys), n_workers=1, buffer_bytes=buffer_bytes)
