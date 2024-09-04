import numpy as np
import multiprocessing as mp

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
        self.it = it
        self.in_queue = DejaQueue(buffer_bytes)
        self.out_queue = DejaQueue(buffer_bytes)
        self.k = mp.Value("l", 0)
        self.k_changed = mp.Condition()
        self.n_workers = n_workers
        self.workers = [mp.Process(target=self._worker_fcn, args=(pid, fcn), kwargs=kwargs) for pid in range(n_workers)]
        [w.start() for w in self.workers]
        self.generator = self._lazymap_generator(it, n_workers)

    def __len__(self):
        return len(self.it)

    def __iter__(self):
        return self.generator
    
    def _worker_fcn(self, pid, fcn, **kwargs):
        while not self.in_queue.done:
            item = self.in_queue.get()
            if item is None: break
            res = fcn(item, **kwargs)
            with self.k_changed:
                self.k_changed.wait_for(lambda: self.k.value % self.n_workers == pid)
                self.out_queue.put(res)
                self.k.value += 1
                self.k_changed.notify_all()

    def _lazymap_generator(self, it, n_workers):
        it = iter(it)
        [self.in_queue.put(next(it)) for i in range(n_workers)]
        k = -1
        for k, item in enumerate(it):
            res = self.out_queue.get()
            self.in_queue.put(item)
            yield res
        for k in range(k + 1, k + 1 + n_workers):
            yield self.out_queue.get()
        self.close()

    def close(self):
        [self.in_queue.put(None) for _ in self.workers]
        [w.join() for w in self.workers]
        self.out_queue = None
        self.in_queue = None

    def compute(self, progress=True, ndarray=True, **kwargs):
        ''' Computes the results of the lazymap and returns them as a list.

        Args:
            progress (bool): whether to show a tqdm progress bar (default: True)
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

class WorkerWrapper:
    ''' A helper class used by the Parallel decorator to wrap a class to make it a parallel worker.

    Args:
        cls (class): the class to be wrapped
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
    '''
    def __init__(self, cls, n_workers=1, buffer_bytes=10e6):
        self.instance = None
        self.cls = cls
        self.n_workers = n_workers
        self.buffer_bytes = int(buffer_bytes)
    def __call__(self, *init_args, **init_kwargs):
        def mapper(iterable, **map_kwargs):
            def worker(arg, **kwargs): 
                if self.instance is None:
                    self.instance = self.cls(*init_args, **init_kwargs)
                return self.instance(arg, **kwargs)
            return lazymap(worker, iterable, self.n_workers, self.buffer_bytes, **map_kwargs)
        return mapper

def Parallel(n_workers=1, buffer_bytes=10e6):
    ''' A wrapper to make a class or callablle a parallel worker. Can be used as a decorator.

    Args:
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
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
