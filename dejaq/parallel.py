import numpy as np
import multiprocessing as mp
from time import sleep
from . import DejaQueue
        

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

    def __init__(self, fcn, it, n_workers=1, buffer_bytes=10e6, **kwargs):
        self._it = it
        self._in_queue = DejaQueue(buffer_bytes)
        self._out_queue = DejaQueue(buffer_bytes)
        self._k = mp.Value("l", 0)
        self._k_changed = mp.Condition()
        self._n_workers = n_workers
        self._workers = [mp.Process(target=self._worker_fcn, args=(pid, fcn), kwargs=kwargs) for pid in range(n_workers)]
        [w.start() for w in self._workers]
        self.generator = self._lazymap_generator(it, n_workers)

    def __len__(self):
        return len(self._it)

    def __iter__(self):
        return self.generator
    
    def _worker_fcn(self, pid, fcn, **kwargs):
        while not self._in_queue.done:
            item = self._in_queue.get()
            if item is None: break
            res = fcn(item, **kwargs)
            with self._k_changed:
                self._k_changed.wait_for(lambda: self._k.value % self._n_workers == pid)
                self._out_queue.put(res)
                self._k.value += 1
                self._k_changed.notify_all()

    def _lazymap_generator(self, it, n_workers):
        it = iter(it)
        [self._in_queue.put(next(it)) for i in range(n_workers)]
        k = -1
        for k, item in enumerate(it):
            res = self._out_queue.get()
            self._in_queue.put(item)
            yield res
        for k in range(k + 1, k + 1 + n_workers):
            yield self._out_queue.get()
        self.close()

    def close(self):
        ''' Sends a termination signal to the workers, waits for them to finish, and deletes the queues.
        ''' 
        [self._in_queue.put(None) for _ in self._workers]
        [w.join() for w in self._workers]
        self._out_queue.close()
        self._in_queue.close()
        while not self._out_queue.done:
            sleep(0.01)
        self._out_queue = None
        self._in_queue = None

    def compute(self, progress=True, ndarray=True, **kwargs):
        ''' Computes the results of the lazymap and returns them as a list or ndarray

        Args:
            progress (bool): whether to show a tqdm progress bar (default: True)
            ndarray (bool): whether to try to return the results as a numpy array (default: True)
            **kwargs: optional, being passed to tqqdm

        Returns:
            (list): a list of the results of fcn(item) for each item in it
        '''
        iterable = self
        if progress:
            from tqdm.auto import tqdm
            iterable = tqdm(iterable)
        out = list(iterable)
        if ndarray:
            try:
                return np.array(out)
            except:
                pass
        else:
            return out



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