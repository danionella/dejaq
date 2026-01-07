import gc, sys, traceback, time, datetime
import logging
logging.basicConfig(level=logging.INFO)

import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import threading
import concurrent.futures

import numpy as np
import cloudpickle
from .queues import DejaQueue, NamedSemaphore

ctx = mp.get_context("spawn")

def passthrough(item):
    return item

def printer(*args, **kwargs):
    print(args, kwargs)




class lazymap:
    ''' Returns an Iterable, functionally related to map (except that outputs are calculated by a pool of processes).
    
    Args:
        fcn (callable): function that is being mapped. Signature: fcn(item, **kwargs)
        it (iterable): iterable that maps over the function, providing items as arguments
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)

        **kwargs: optional, being passed to fcn

    Returns: 
        (iterable): an iterable that returns the results of fcn(item) for each item in it
    '''

    def __init__(self, fcn, it, n_workers=1, buffer_bytes=10e6, start_type='lazy', **kwargs):
        if n_workers > 1 and not isinstance(it, lazymap):
            it = lazymap(None, it, n_workers=1, buffer_bytes=buffer_bytes) # to make sure that the input iterable is thread-safe  
        self._it = it
        self._start_type = start_type
        self._out_queue = DejaQueue(buffer_bytes)
        self._in_sem = [NamedSemaphore() for _ in range(n_workers)]
        self._out_sem = [NamedSemaphore() for _ in range(n_workers)]
        if start_type == "eager":
            self._in_sem[0].release()
            self._started = True
        else:
            self._started = False
        self._out_sem[0].release()
        self._n_workers = n_workers
        pkl = cloudpickle.dumps((fcn, self._it))
        worker_args = [pkl, self._out_queue, n_workers, self._in_sem, self._out_sem]
        _workers = [ctx.Process(target=self._worker_fcn, args=(pid, *worker_args), kwargs=kwargs) for pid in range(n_workers)]
        [w.start() for w in _workers]
        self.pids = [w.pid for w in _workers]

    def __len__(self):
        return len(self._it)

    def __iter__(self):
        """ Returns an iterator over the results of fcn(item) for each item in it """
        if self._start_type == "lazy" and not self._started:
            self._in_sem[0].release()
            self._started = True
        return self._lazymap_generator(self._out_queue)

    @staticmethod
    def _worker_fcn(pid, pkl, _out_queue, _n_workers, in_sem, out_sem, **kwargs):
        import cloudpickle, traceback
        fcn, it = cloudpickle.loads(pkl)
        nxt = (pid + 1) % _n_workers
        in_sem[pid].acquire()
        iterator = iter(it)
        in_sem[pid].release()

        while True:
            in_sem[pid].acquire()
            try:
                item = next(iterator)
            except StopIteration:
                in_sem[nxt].release()
                break
            in_sem[nxt].release()
            try:
                res = item if fcn is None else fcn(item, **kwargs)
            except Exception as e:
                tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                res = {"type":"exception","exc_type":str(type(e)), "exc_msg":str(e), "traceback":tb}
            out_sem[pid].acquire()
            _out_queue.put(res)
            out_sem[nxt].release()

        out_sem[pid].acquire()
        _out_queue._signal_stop()
        out_sem[nxt].release()

    @staticmethod
    def _lazymap_generator(_out_queue):
        for item in _out_queue:
            if isinstance(item, dict) and item.get("type") == "exception":
                raise RuntimeError(f"Exception in worker: {item['exc_type']}: {item['exc_msg']}\nTraceback:\n{item['traceback']}")
            yield item
        _out_queue._signal_stop() # in case multiple workers, make sure that all are stopped

    def close(self):
        ''' Sends a termination signal to the workers, waits for them to finish, and deletes the queues.
        ''' 
        self._out_queue.close()
        while not self._out_queue.done:
            time.sleep(0.01)
        for w in self._workers:
            w.join(timeout=.2)
        self._out_queue = None

    def compute(self, progress=True, ndarray=True, tqdm_args={}):
        ''' Computes the results of the lazymap and returns them as a list or ndarray

        Args:
            progress (bool): whether to show a tqdm progress bar (default: True)
            ndarray (bool): whether to try to return the results as a numpy array (default: True)
            tqdm_args: 

        Returns:
            (list): a list of the results of fcn(item) for each item in it
        '''
        iterable = self
        if progress:
            from tqdm.auto import tqdm
            iterable = tqdm(iterable, **tqdm_args)
        out = list(iterable)
        if ndarray:
            try:
                return np.array(out)
            except:
                return out
        else:
            return out

    def submit(self):
        ''' Submits the computation to a background thread and returns a Future object.

        Returns:
            (Future): a Future object that can be used to fetch the result later
        '''
        future = concurrent.futures.Future()
        def run():
            try:
                result = self.compute(tqdm_args=dict(disable=True))
                future.set_result(result)
            except Exception as e:
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
        if not self._started:
            self._in_sem[0].release()
            self._started = True
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
    return [lazymap(None, QueueIterator(q), n_workers=1) for q in queues]

def zip(*lazys):
    """A helper to zip multiple lazymap iterables into a single lazymap iterable of tuples.
    Args:
        *lazys (lazymap): the lazymap iterables to zip
    """
    def zipper(items):
        return tuple(items)
    return lazymap(zipper, zip(*lazys), n_workers=1)