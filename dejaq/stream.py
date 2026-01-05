import multiprocessing as mp
from multiprocessing import shared_memory
import logging
import sys
import abc
import time
import psutil

logging.basicConfig(level=logging.INFO, format="%(processName)s %(levelname)s: %(message)s", 
    stream=sys.stdout, force=True)

import cloudpickle
import numpy as np

from .queues import NamedSemaphore, DejaQueue

class _SENTINEL:
    """A sentinel object used to signal termination in queues."""
    pass


class Counter:
    """A simple counter iterator that yields increasing integers starting from 0."""
    def __init__(self):
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        val = self.count
        self.count += 1
        return val


class CurrentTime:
    """An iterator that yields the current time in seconds since the epoch."""
    def __iter__(self):
        return self

    def __next__(self):
        return time.time()


class RateLimiter:
    """A rate limiter that enforces a minimum interval between calls."""
    def __init__(self, rate):
        """
        Args:
            rate (float): Maximum rate in Hz (calls per second).
        """
        self.interval = 1.0 / rate
        self.next_time = 0
    
    def wait(self):
        """Wait until the next allowed call time."""
        now = time.time()
        if now < self.next_time:
            time.sleep(self.next_time - now)
        self.next_time = max(self.next_time, now) + self.interval
    
    def __call__(self, func):
        """Decorator to rate-limit a function."""
        def wrapper(*args, **kwargs):
            self.wait()
            return func(*args, **kwargs)
        return wrapper


class RateLimitedIterator:
    """Wraps an iterator to enforce a maximum consumption rate."""
    def __init__(self, it, rate):
        self._it = iter(it)
        self._limiter = RateLimiter(rate)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self._limiter.wait()
        return next(self._it)


class BaseNode(abc.ABC):
    """Base class for all stream nodes."""

    _mapped = False

    def map(self,  **kwargs):
        assert not self._mapped, "Node has already been mapped"
        self._mapped = True
        return MapNode(self, **kwargs)

    def tee(self, count=2, buffer_bytes=10e6):
        return tee(self, count=count, buffer_bytes=buffer_bytes)

    def zip(self, *nodes):
        for node in nodes:
            assert not node._mapped, f"Node {node} has already been mapped"
        return ZipNode(self, *nodes)
    
    def sink(self, *args, **kwargs):
        """Creates a sink node that applies a function to each item in the stream.
        Args:
            **kwargs: being passed to MapNode

        Returns:
            (MapNode): a MapNode configured as a sink node
        """
        return MapNode(self, *args, sink=True, **kwargs)

    def run(self, progress=True, keep_outputs=True, ndarray=False, **kwargs):
        """Run the pipeline.

        By default this collects results into memory (similar to the old
        ``compute()`` method). For sink-style pipelines (write to disk, publish
        to network, etc.) set ``keep_outputs=False`` or use :meth:`foreach`.

        Args:
            progress (bool): whether to show a tqdm progress bar (default: True)
            keep_outputs (bool): whether to collect and return outputs (default: True)
            ndarray (bool): whether to try to return the results as a numpy array (default: True)
            **kwargs: optional, being passed to tqqdm

        Returns:
            Union[list, numpy.ndarray, int]:
                - if ``keep_outputs=True``: list or ndarray of outputs
                - if ``keep_outputs=False``: number of items consumed
        """
        iterable = self
        if progress:
            from tqdm.auto import tqdm

            iterable = tqdm(iterable, **kwargs)

        if not keep_outputs:
            n = 0
            for _ in iterable:
                n += 1
            return n

        out = list(iterable)
        if ndarray:
            try:
                return np.array(out)
            except:
                return out
        return out


class ZipNode(BaseNode):
    """ Zips multiple nodes together into a single node yielding tuples of items from each node. """
    def __init__(self, *nodes):
        self._nodes = nodes

    def __iter__(self):
        yield from zip(*self._nodes)


class _DummyQueue:
    """A dummy queue that does nothing. Used for sink nodes."""
    def put(self, item):
        pass
    def close(self):
        pass


class MapNode(BaseNode):
    """Returns a map over a function using one or more separate processes, yielding results in order.

    Args:
        it (iterable): iterable over which fcn is being mapped
        fcn (callable): function that is being mapped. Signature: fcn(item, **kwargs)
        factory (callable): class or function that returns an instance. Signature: factory() -> obj
        call_fcn (callable): if factory is provided, method to call on the instance for each item. Signature: call_fcn(obj, item) -> result (e.g. lambda obj, item: obj.process(item))
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
        start_mode (str): 'eager' (default), 'lazy' or 'manual', determining when the workers start processing.
        **kwargs: optional, being passed to fcn

    Returns:
        (iterable): an iterable that returns the results of fcn(item) for each item in it
    """

    def __init__(self, it, fcn=None, factory=None, call_fcn=lambda obj, item: obj(item), 
                 n_workers=1, buffer_bytes=10e6, start_mode="eager", sink=False, **kwargs):
        assert (fcn is None) != (factory is None), "Either fcn or factory must be provided, but not both."
        self._it = it
        self._n_workers = n_workers
        self._sink = sink
        self._out_queue = DejaQueue(buffer_bytes) if not sink else _DummyQueue()
        self._input_k = mp.Value("l", 0)   # For grabbing items in order
        self._output_k = mp.Value("l", 0)  # For outputting results in order
        self._start_event = mp.Event()
        self._cancel_event = mp.Event()
        self._input_cond = mp.Condition()
        self._output_cond = mp.Condition()
        self._finished = mp.Event()
        pkl = cloudpickle.dumps((fcn, factory, call_fcn))
        _workers = [mp.Process(target=self._worker_fcn, args=(wid, pkl, it), kwargs=kwargs, daemon=True) 
                    for wid in range(n_workers)]
        [w.start() for w in _workers]
        self._pids = [w.pid for w in _workers]
        self._start_mode = start_mode
        if start_mode == 'eager':
            self._start_event.set()

    def __len__(self):
        return len(self._it)

    def __iter__(self):
        if self._sink:
            raise TypeError("Sink nodes are terminal and cannot be iterated over.")
        return self._lazymap_generator()

    def _worker_fcn(self, wid, fcn_pkl, it, **kwargs):
        fcn, factory, call_fcn = cloudpickle.loads(fcn_pkl)
        instance = None
        try: 
            if factory is not None:
                instance = factory()
                fcn = lambda item: call_fcn(instance, item)
        except Exception as e:
            logging.error("Exception during worker %d initialization: %s", wid, e, exc_info=True)
            self._signal_done()
            return

        try:
            self._start_event.wait()
            iterator = iter(it)
            while True:
                if self._cancel_event.is_set():
                    break

                # Wait for my turn to grab next item
                with self._input_cond:
                    self._input_cond.wait_for(lambda: self._input_k.value % self._n_workers == wid or self._cancel_event.is_set())
                    if self._cancel_event.is_set():
                        break
                    try:
                        item = next(iterator)
                        my_idx = self._input_k.value
                        self._input_k.value += 1
                    except StopIteration:
                        break
                    finally:
                        self._input_cond.notify_all()

                if self._cancel_event.is_set():
                    break

                res = fcn(item, **kwargs)
                logging.debug(f"Worker {wid} processed item {my_idx} with result: {res}")

                # Wait for my turn to output
                with self._output_cond:
                    self._output_cond.wait_for(lambda: self._output_k.value == my_idx or self._cancel_event.is_set())
                    if self._cancel_event.is_set():
                        break
                    self._out_queue.put(res)
                    self._output_k.value += 1
                    self._output_cond.notify_all()

        except Exception:
            logging.error("Exception in worker %d", wid, exc_info=True)
        finally:
            logging.debug(f"Worker {wid} done.")
            self._signal_done()
            if instance is not None:
                try: 
                    instance.close()
                except:
                    pass

    def _signal_done(self):
        """Signal that a worker has finished. Send sentinel once and wake other workers."""
        self._cancel_event.set()
        with self._input_cond:
            self._input_cond.notify_all()
        with self._output_cond:
            if not self._finished.is_set():
                self._finished.set()
                self._out_queue.put(_SENTINEL)
            self._output_cond.notify_all()

    def _lazymap_generator(self):
        res = None
        if self._start_mode == 'lazy':
            self._start_event.set()
        while True:
            res = self._out_queue.get()
            if res is _SENTINEL:
                break
            yield res
        self.close()

    def start(self):
        """Starts the map operation, signaling all workers to begin processing."""
        self._start_event.set()

    def stop(self):
        """Stops the map operation, signaling all workers to stop processing and exit."""
        self._cancel_event.set()
        with self._input_cond:
            self._input_cond.notify_all()
        with self._output_cond:
            self._output_cond.notify_all()

    def close(self):
        """Sends a termination signal to the workers, waits for them to finish, and deletes the queues."""
        self._out_queue.close()

    def is_running(self) -> bool:
        """Checks if worker processes are still running (not zombie or terminated).

        Returns:
            bool: True if all workers are alive and not zombies, False otherwise.
        """
        try:
            for pid in self._pids:
                proc = psutil.Process(pid)
                if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                    return False
            return True
        except psutil.NoSuchProcess:
            return False

    def wait(self, timeout=None):
        """Waits for all worker processes to finish.

        Args:
            timeout (float): maximum time to wait in seconds. If None, waits indefinitely.
        """
        start_time = time.time()
        while self.is_running():
            if timeout is not None and (time.time() - start_time) > timeout:
                break
            time.sleep(0.1)


def Source(it=None, fcn=None, factory=None, call_fcn=lambda obj: obj(), 
                 n_workers=1, buffer_bytes=10e6, rate=None, **kwargs):
    """Returns a source Node that yields items from an iterable. Either a callable function, an instance factory, or iterator must be provided. 
    If none are provided, a ManualSource is returned.

    Args:
        it (iterable): iterable to yield items from
        fcn (callable): function that returns items. Signature: fcn(**kwargs) -> item
        factory (class or callable): class or instance factory function that returns an instance. Signature: factory() -> obj
        call_fcn (callable): if factory is provided, method to call on the instance for each item. Signature: call_fcn(obj) -> item (e.g. lambda obj: obj.get_frame())
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
        rate (float): rate limit in Hz (items per second). If None, no rate limiting is applied.

    Returns:
        (MapNode): a MapNode configured as a source node
    """

    nargs = (fcn is not None) + (factory is not None) + (it is not None)
    assert nargs <= 1, "Either fcn, factory, or it must be provided, or none, but not more than one."

    if nargs == 0:
        return ManualSource(buffer_bytes=buffer_bytes)

    # Apply rate limiting if specified
    if rate is not None:
        limiter = RateLimiter(rate)
        if it is not None:
            it = RateLimitedIterator(it, rate)
        elif fcn is not None:
            original_fcn = fcn
            fcn = lambda **kw: (limiter.wait(), original_fcn(**kw))[1]
        elif factory is not None:
            original_call_fcn = call_fcn
            call_fcn = lambda obj: (limiter.wait(), original_call_fcn(obj))[1]

    if it is not None:
        assert n_workers == 1, "Only supports n_workers=1 when arg is an iterable"
        return MapNode(it=it, fcn=lambda x: x, n_workers=1, buffer_bytes=buffer_bytes, start_mode="manual")
    elif fcn is not None:
        return MapNode(it=Counter(), fcn=lambda x: fcn(**kwargs), 
                            n_workers=n_workers, buffer_bytes=buffer_bytes, start_mode="manual")
    elif factory is not None:
        return MapNode(it=Counter(), factory=factory, call_fcn=lambda obj, item: call_fcn(obj),
                            n_workers=n_workers, buffer_bytes=buffer_bytes, start_mode="manual")


class ManualSource(MapNode):
    """A source Node where items can be manually put into the stream.
    
    Args:
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
    """

    def __init__(self, buffer_bytes=10e6):
        super().__init__(fcn=lambda x: x, it=Counter(), n_workers=0, buffer_bytes=buffer_bytes, start_mode="manual")
    def put(self, item):
        """Puts an item into the source stream."""
        self._out_queue.put(item)
    def stop(self):
        """Stops the source stream."""
        self._out_queue.put(_SENTINEL)


def _tee_worker(it, queues, cancel_event):
    try:
        for item in it:
            if cancel_event.is_set():
                break
            for q in queues:
                q.put(item)
    finally:
        for q in queues:
            q.put(_SENTINEL)

class _TeeOutputNode(BaseNode):

    def __init__(self, queue, cancel_event, pid):
        super().__init__()
        self._queue = queue
        self._pid = pid
        self._cancel_event = cancel_event

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            yield item

    def cancel(self):
        self._cancel_event.set()

def tee(it, count=2, buffer_bytes=10e6):
    """Split an iterable into multiple independent Node streams.
    
    Args:
        it (iterable): input iterable
        count (int): number of output streams (default: 2)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)

    Returns:
        (tuple): tuple of output Node streams
    """
    queues = [DejaQueue(buffer_bytes) for _ in range(count)]
    cancel_event = mp.Event()

    p = mp.Process(target=_tee_worker, args=(it, queues, cancel_event))
    p.start()

    return tuple(_TeeOutputNode(q, cancel_event, p.pid) for q in queues)
