import multiprocessing as mp
import os
import sys
import abc
import threading
import time
import concurrent
import psutil
import queue as _queue 
import logging

import cloudpickle
import numpy as np
from tqdm.auto import tqdm

from .queues import NamedSemaphore, DejaQueue

logging.basicConfig(
    level=logging.INFO, format="%(processName)s %(levelname)s: %(message)s", stream=sys.stdout, force=True
)

_mp_ctx = mp.get_context("spawn")

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

    def __init__(self):
        self._mapped = False
        self._iterated = False

    def _retain(self, *nodes):
        """Keep downstream nodes alive as long as this node is referenced."""
        if not nodes:
            return
        if not hasattr(self, "_children"):
            self._children = []
        self._children.extend([n for n in nodes if n is not None])

    def map(self, fcn=None, cls=None, start_mode='lazy', cls_fcn=lambda obj, item: obj(item), init_kwargs=None, **kwargs):
        """Creates a map node that applies a function or class method to each item in the stream, depending on the type of arg provided.

        Args:
            fcn (callable): function that is being mapped (Signature: fcn(item))
            cls (class or callable): class/factory to instantiate (Signature: cls(**init_kwargs) -> obj)
            start_mode (str): 'eager', 'lazy' or 'manual', determining when the workers start processing.
            cls_fcn (callable): if arg is a class or factory, method to call on the instance for each item. Signature: cls_fcn(obj, item)
            init_kwargs (dict): keyword arguments to pass to the class constructor
            **kwargs: optional, being passed to MapNode.__init__()
        """
        assert not self._mapped, "Node has already been mapped"
        self._mapped = True

        assert (fcn is not None) != (cls is not None), "Either fcn or cls must be provided, but not both."

        # check if arg is class or class factory:
        if cls is not None:
            assert type(cls) == type or callable(cls), "cls must be a class or a callable factory function with signature cls(**init_kwargs)."
            node = MapNode.from_class(cls=cls, cls_fcn=cls_fcn, it=self, init_kwargs=init_kwargs, **kwargs)
            self._retain(node)
            return node
        elif fcn is not None:
            assert callable(fcn), "fcn must be a callable function with signature fcn(item)."
            node = MapNode(it=self, fcn=fcn, start_mode=start_mode, **kwargs)
            self._retain(node)
            return node

    def map_class(self, cls=None, cls_fcn=lambda obj, item: obj(item), init_kwargs=None, **map_kwargs):
        """Create a MapNode that instantiates cls once and calls cls_method on it for each item.
        Args:
            cls (class or callable): class or class factory. Signature: cls(**init_kwargs) -> obj
            cls_fcn (callable): method of the class to call for each item. Signature cls_fcn(obj, item)
            init_kwargs (dict): keyword arguments to pass to the class constructor
            **map_kwargs: optional, being passed to MapNode.__init__()
        """
        assert not self._mapped, "Node has already been mapped"
        self._mapped = True
        node = MapNode.from_class(cls=cls, cls_fcn=cls_fcn, it=self, init_kwargs=init_kwargs, **map_kwargs)
        self._retain(node)
        return node

    def tee(self, count=2, buffer_bytes=10e6):
        """Splits the stream into multiple independent streams.
        
        Args:
            count (int): number of output streams (default: 2)
            buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
        
        Returns:
            (tuple): tuple of output Node streams
        """
        assert not self._mapped, "Node has already been mapped"
        self._mapped = True
        nodes = get_tee(self, count=count, buffer_bytes=buffer_bytes)
        self._retain(*nodes)
        return nodes

    def zip(self, *nodes, mode='sync'):
        """Zips multiple nodes together into a single node yielding tuples of items from each node.
        Args:
            *nodes (BaseNode): nodes to zip with this node
            mode (str): One of "buffer", "latest", or "sync"
                - "buffer": collects all items from secondary nodes since last emission as tuples
                - "latest": keeps only the most recent item from each secondary node (None initially)
                - "sync": waits for one new item from each secondary node (like standard zip)
        """
        for node in [self, *nodes]:
            assert not node._mapped, f"Node {node} has already been mapped"
        for node in [self, *nodes]:
            node._mapped = True
        out = ZipNode(self, *nodes, mode=mode)
        self._retain(out)
        return out

    def tqdm(self, **tqdm_kwargs):
        """Wraps the node with a tqdm progress bar.

        Args:
            **tqdm_kwargs: being passed to tqdm.auto.tqdm

        Returns:
            (tqdm.auto.tqdm): a tqdm-wrapped iterable
        """
        return self.map_class(cls=TqdmProxy, init_kwargs=tqdm_kwargs)

    def sink(self, *args, **kwargs):
        """Creates a sink node that applies a function to each item in the stream.
        Args:
            **kwargs: being passed to MapNode

        Returns:
            (MapNode): a MapNode configured as a sink node
        """
        assert not self._iterated, "Node has already been iterated"
        self._iterated = True
        assert not self._mapped, "Node has already been mapped"
        node = MapNode(self, *args, sink=True, start_mode='eager', **kwargs)
        self._retain(node)
        return node

    def run(self, *args, **kwargs):
        """Alias for :meth:`compute`."""
        return self.compute(*args, **kwargs)

    def compute(self, progress=True, keep_outputs=True, ndarray=False, **kwargs):
        """Run the pipeline and collect the results.

        By default this collects results into memory (similar to the old
        ``compute()`` method). For sink-style pipelines (write to disk, publish
        to network, etc.) set ``keep_outputs=False`` or use :meth:`foreach`.

        Args:
            progress (bool): whether to show a tqdm progress bar (default: True)
            keep_outputs (bool): whether to collect and return outputs (default: True)
            ndarray (bool): whether to try to return the results as a numpy array (default: True)
            **kwargs: optional, being passed to tqdm

        Returns:
            Union[list, numpy.ndarray, int]:
                - if ``keep_outputs=True``: list or ndarray of outputs
                - if ``keep_outputs=False``: number of items consumed
        """
        assert not self._iterated, "Node has already been iterated"
        self._iterated = True
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
            except Exception:
                return out
        return out

    def submit(self, keep_outputs=True, ndarray=False):
        """Submits the computation to a background thread and returns a Future object.

        Args:
            keep_outputs (bool): whether to collect outputs (default: True) or just count items
            ndarray (bool): whether to try to return the results as a numpy array (default: True)

        Returns:
            (Future): a Future object that can be used to fetch the result later
        """
        assert not self._iterated, "Node has already been iterated"
        future = concurrent.futures.Future()

        def run():
            try:
                result = self.run(progress=False, keep_outputs=keep_outputs, ndarray=ndarray)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return future


# class ZipNode(BaseNode):
#     """ Zips multiple nodes together into a single node yielding tuples of items from each node. """
#     def __init__(self, *nodes):
#         self._nodes = nodes

#     def __iter__(self):
#         yield from zip(*self._nodes)


class _DummyQueue:
    """A dummy queue that does nothing. Used for sink nodes."""
    def put(self, item):
        pass
    def close(self):
        pass


class MapNode(BaseNode):
    """Returns an Iterable, functionally related to map (except that outputs are calculated by a pool of processes).

    Args:
        fcn (callable): function that is being mapped. Signature: fcn(item, **kwargs)
        it (iterable): iterable that maps over the function, providing items as arguments
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)

        **kwargs: optional, being passed to fcn

    Returns:
        (iterable): an iterable that returns the results of fcn(item) for each item in it
    """

    def __init__(self, it, fcn=lambda x: x, n_workers=1, buffer_bytes=10e6, start_mode="lazy", sink=False, **kwargs):
        super().__init__()
        self._it = it
        self._start_mode = start_mode
        self._sink = sink
        self._out_queue = DejaQueue(buffer_bytes) if not sink else _DummyQueue()
        self._in_sem = [NamedSemaphore() for _ in range(n_workers)]
        self._out_sem = [NamedSemaphore() for _ in range(n_workers)]
        if n_workers > 0:
            self._in_sem[0].release()
            self._out_sem[0].release()
        self._start_event = _mp_ctx.Event()
        self._cancel_event = _mp_ctx.Event()
        self._n_workers = n_workers
        pkl = cloudpickle.dumps(fcn)
        worker_args = [pkl, self._out_queue, n_workers, self._in_sem, self._out_sem, self._cancel_event]
        _workers = [
            _mp_ctx.Process(target=self._worker_fcn, args=(pid, *worker_args), kwargs=kwargs, daemon=True)
            for pid in range(n_workers)
        ]
        [w.start() for w in _workers]
        self._pids = [w.pid for w in _workers]
        if self._start_mode == "eager":
            self._start_event.set()

    def __len__(self):
        return len(self._it)

    def __iter__(self):
        """Returns an iterator over the results of fcn(item) for each item in it"""
        if self._sink:
            raise TypeError("Sink nodes are terminal and cannot be iterated over.")
        if self._start_mode == "lazy":
            self._start_event.set()
        return self._lazymap_generator()

    def _worker_fcn(self, wid, pkl, _out_queue, _n_workers, in_sem, out_sem, cancel_event, **kwargs):
        import cloudpickle, traceback

        fcn = cloudpickle.loads(pkl)
        nxt = (wid + 1) % _n_workers
        self._start_event.wait()
        logging.info(f"Map worker (#: {wid}, PID: {os.getpid()}, input: {self._it}) started.")
        in_sem[wid].acquire()
        iterator = iter(self._it)
        in_sem[wid].release()

        while True:
            in_sem[wid].acquire()
            if cancel_event.is_set():
                in_sem[nxt].release()
                break
            try:
                item = next(iterator)
            except StopIteration:
                in_sem[nxt].release()
                break
            in_sem[nxt].release()

            if cancel_event.is_set():
                break
            try:
                res = item if fcn is None else fcn(item, **kwargs)
            except Exception as e:
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                res = {"type": "exception", "exc_type": str(type(e)), "exc_msg": str(e), "traceback": tb}
            out_sem[wid].acquire()
            try:
                if cancel_event.is_set():
                    break
                _out_queue.put(res)
            finally:
                out_sem[nxt].release()

        out_sem[wid].acquire()
        _out_queue.put(_SENTINEL)
        out_sem[nxt].release()
        logging.info(f"Map worker (#: {wid}, PID: {os.getpid()}, input: {self._it}) done.")

    def stop(self):
        """Stop worker processes.

        This is cooperative cancellation: workers will exit once they reach a cancellation
        check point. If the upstream iterator blocks indefinitely, workers may not be able
        to stop until that iterator unblocks.
        """
        if getattr(self, "_cancel_event", None) is None:
            return

        self._cancel_event.set()
        self._start_event.set()

        try:
            self._out_queue.put(_SENTINEL)
        except Exception:
            pass

        for sem in getattr(self, "_in_sem", []):
            try:
                sem.release()
            except Exception:
                pass
        for sem in getattr(self, "_out_sem", []):
            try:
                sem.release()
            except Exception:
                pass

    def _lazymap_generator(self):
        res = None
        while True:
            item = self._out_queue.get()
            if isinstance(item, dict) and item.get("type") == "exception":
                self._out_queue.put(_SENTINEL)
                logging.error("Exception in worker: %s: %s\nTraceback:\n%s", item['exc_type'], item['exc_msg'], item['traceback'])
                raise RuntimeError(
                    f"Exception in worker: {item['exc_type']}: {item['exc_msg']}\nTraceback:\n{item['traceback']}"
                )
            elif item is _SENTINEL:
                break
            yield item
        self._out_queue.put(_SENTINEL)

    def close(self, timeout: float = 1.0, terminate: bool = True):
        """Stop workers and close the output queue (PID-based; keeps pickling friendly)."""
        self.stop()

        pids = getattr(self, "_pids", []) or []
        procs = []
        for pid in pids:
            try:
                procs.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                pass

        gone, alive = psutil.wait_procs(procs, timeout=timeout)
        if terminate and alive:
            for p in alive:
                try:
                    p.terminate()
                except Exception:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=0.5)
            for p in alive:
                try:
                    p.kill()
                except Exception:
                    pass

        try:
            self._out_queue.close()
        except Exception:
            pass
        self._out_queue = None

    def start(self):
        """Starts the map operation, signaling all workers to begin processing."""
        self._start_event.set()

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

    @staticmethod
    def from_class(cls=None, cls_fcn=None, it=None, init_kwargs=None, **kwargs):
        """Create a MapNode that instantiates cls once and calls cls_method on it for each item.
        Args:
            cls (class or callable): class or class factory to instantiate. Signature: cls(**init_kwargs) -> obj
            cls_fcn (callable): method of the class to call for each item. Signature cls_fcn(obj, item)
            it (iterable): iterable to provide items.
            init_kwargs (dict): keyword arguments to pass to the class constructor
            **kwargs: optional, being passed to MapNode().__init__()
        """
        cls = cls
        instance = None
        init_kwargs = init_kwargs or {}

        def wrapper(item):
            nonlocal instance
            if instance is None:
                instance = cls(**init_kwargs)
            return cls_fcn(instance, item)
        return MapNode(it=it, fcn=wrapper, **kwargs)


class NodeProxy:
    """A proxy that creates an instance from a factory and calls a method on it for each item."""
    def __init__(self, factory, fcn=lambda obj, item: obj(item)):
        self.instance = factory()
        self.fcn = fcn

    def __call__(self, item):
        return self.fcn(self.instance, item)


def Source(it=None, fcn=None, cls=None, call_fcn=lambda obj: obj(), init_kwargs=None,
                 n_workers=1, buffer_bytes=10e6, rate=None, start_mode="lazy", **kwargs):
    """Returns a source dejaq.MapNode. Either an iterator, a callable function or an instance factory must be provided.
    If none are provided, a dejaq.stream.ManualSource is returned.

    Args:
        it (iterable): iterable to yield items from
        fcn (callable): function that returns items. Signature: fcn(**kwargs) -> item
        cls (class or callable): class or instance factory function that returns an instance. Signature: cls() -> obj
        call_fcn (callable): if factory is provided, method to call on the instance for each item. Signature: call_fcn(obj) -> item (e.g. lambda obj, item: obj.get_frame(item))
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
        rate (float): rate limit in Hz (items per second). If None, no rate limiting is applied.
        start_mode (str): 'lazy' (default), 'eager' or 'manual', determining when the workers start processing.
        **kwargs: optional, being passed to fcn

    Returns:
        (MapNode): a MapNode configured as a source node
    """

    nargs = (fcn is not None) + (cls is not None) + (it is not None)
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
        elif cls is not None:
            original_call_fcn = call_fcn
            call_fcn = lambda *args, **kwargs: (limiter.wait(), original_call_fcn(*args, **kwargs))[1]

    if it is not None:
        assert n_workers == 1, "Only supports n_workers=1 when arg is an iterable"
        return MapNode(it=it, fcn=lambda x: x, n_workers=1, buffer_bytes=buffer_bytes, start_mode=start_mode)
    elif fcn is not None:
        return MapNode(it=Counter(), fcn=lambda x: fcn(**kwargs), 
                            n_workers=n_workers, buffer_bytes=buffer_bytes, start_mode=start_mode)
    elif cls is not None:
        cls_fcn = lambda obj, item: call_fcn(obj)
        init_kwargs = init_kwargs or {}
        return MapNode.from_class(it=Counter(), cls=cls, cls_fcn=cls_fcn, init_kwargs=init_kwargs,
                            n_workers=n_workers, buffer_bytes=buffer_bytes, start_mode=start_mode)


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


def _tee_worker(it, queues, cancel_event, start_events):
    for k, start_event in enumerate(start_events):
        logging.info(f"Waiting for start event {k}")
        start_event.wait()
        logging.info(f"Start event {k} received")
    logging.info(f"Tee worker (PID: {os.getpid()}, input: {it}) started.")
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

    def __init__(self, queue, cancel_event, pid, start_event):
        super().__init__()
        self._queue = queue
        self._pid = pid
        self._cancel_event = cancel_event
        self._start_event = start_event
    def __iter__(self):
        self._start_event.set()
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            yield item

    def cancel(self):
        self._cancel_event.set()

def get_tee(it, count=2, buffer_bytes=10e6):
    """Split an iterable into multiple independent Node streams.
    
    Args:
        it (iterable): input iterable
        count (int): number of output streams (default: 2)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)

    Returns:
        (tuple): tuple of output Node streams
    """
    start_events = [_mp_ctx.Event() for _ in range(count)]
    queues = [DejaQueue(buffer_bytes) for _ in range(count)]
    cancel_event = _mp_ctx.Event()

    p = _mp_ctx.Process(target=_tee_worker, args=(it, queues, cancel_event, start_events), daemon=True)
    p.start()

    return tuple(_TeeOutputNode(queues[i], cancel_event, p.pid, start_events[i]) for i in range(count))


class ZipNode(BaseNode):
    """Zips multiple nodes where the first node drives the output rate.

    When the primary (first) node yields an item, an output tuple is produced containing:
    - The item from the primary node
    - Items from secondary nodes, whose format depends on the mode

    Args:
        primary_node (BaseNode): The primary node that drives output timing
        *secondary_nodes (BaseNode): Secondary nodes to zip with the primary
        mode (str): One of "buffer", "latest", or "sync"
            - "buffer": collects all items from secondary nodes since last emission as tuples
            - "latest": keeps only the most recent item from each secondary node (None initially)
            - "sync": waits for one new item from each secondary node (like standard zip)
        maxlen (int): Maximum buffer length per secondary node in buffer mode (default: None, unbounded)
    """

    def __init__(self, primary_node, *secondary_nodes, mode="sync", maxlen=None):
        super().__init__()
        assert mode in ("buffer", "latest", "sync"), "mode must be 'buffer', 'latest', or 'sync'"
        self._primary = primary_node
        self._secondaries = secondary_nodes
        self._mode = mode
        self._maxlen = maxlen

    def __iter__(self):
        if self._mode == "sync":
            yield from zip(self._primary, *self._secondaries)
            return

        start_flag = threading.Event()
        stop_flags = [threading.Event() for _ in self._secondaries]

        try:
            if self._mode == "buffer":
                queues = [_queue.Queue(maxsize=self._maxlen or 0) for _ in self._secondaries]
                yield from self._iter_buffer(stop_flags, queues, start_flag)
            else:  # latest
                queues = [_queue.Queue() for _ in self._secondaries]
                yield from self._iter_latest(stop_flags, queues, start_flag)
        finally:
            start_flag.set()
            for flag in stop_flags:
                flag.set()

    @staticmethod
    def _collect_items_buffer(node, q, stop_flag, start_flag):
        """Worker function to collect items from a secondary node into a queue."""
        try:
            start_flag.wait()
            logging.info(f"Zip thread (PID: {os.getpid()}, input: {node}) started.")
            for item in node:
                if stop_flag.is_set():
                    break
                try:
                    q.put_nowait(item)
                except _queue.Full:
                    pass  # Queue full, skip item if maxlen is set
        except Exception as e:
            logging.error(f"Error in secondary node: {e}")
        logging.info(f"Zip thread (PID: {os.getpid()}, input: {node}) done.")

    @staticmethod
    def _update_latest(node, q, stop_flag, start_flag):
        """Worker function to update the latest item in queue."""
        try:
            start_flag.wait()
            logging.info(f"Zip thread (PID: {os.getpid()}, input: {node}) started.")
            for item in node:
                if stop_flag.is_set():
                    break
                # Always put new item, draining any old one first
                while True:
                    try:
                        q.get_nowait()
                    except _queue.Empty:
                        break
                q.put(item)
        except Exception as e:
            logging.error(f"Error in secondary node: {e}")
        logging.info(f"Zip thread (PID: {os.getpid()}, input: {node}) done.")

    def _iter_buffer(self, stop_flags, queues, start_flag):
        """Buffer mode: collect all items from secondaries between primary emissions."""
        n_secondaries = len(self._secondaries)

        threads = [threading.Thread(target=self._collect_items_buffer, args=(node, queues[i], stop_flags[i], start_flag), daemon=True)
            for i, node in enumerate(self._secondaries)]
        [t.start() for t in threads]

        # Let worker threads begin iterating their inputs only after all threads exist.
        start_flag.set()

        output = [None] * (1 + n_secondaries)

        for primary_item in self._primary:
            output[0] = primary_item
            for i in range(n_secondaries):
                items = []
                while True:
                    try:
                        items.append(queues[i].get_nowait())
                    except _queue.Empty:
                        break
                output[i + 1] = tuple(items)
            yield tuple(output)

    def _iter_latest(self, stop_flags, queues, start_flag):
        """Latest mode: keep only the most recent item from each secondary."""
        n_secondaries = len(self._secondaries)
        last_seen = [None] * n_secondaries

        threads = [threading.Thread(target=self._update_latest, args=(node, queues[i], stop_flags[i], start_flag), daemon=True)
            for i, node in enumerate(self._secondaries)]
        [t.start() for t in threads]

        # Let worker threads begin iterating their inputs only after all threads exist.
        start_flag.set()

        output = [None] * (1 + n_secondaries)

        for primary_item in self._primary:
            output[0] = primary_item
            for i in range(n_secondaries):
                while True:
                    try:
                        last_seen[i] = queues[i].get_nowait()
                    except _queue.Empty:
                        break
                output[i + 1] = last_seen[i]
            yield tuple(output)

class TqdmProxy:
    """A node that wraps an iterable with a tqdm progress bar.

    Args:
        it (iterable): iterable to wrap
        **kwargs: keyword arguments being passed to tqdm
    """

    def __init__(self, **kwargs):
        self.pbar = tqdm(**kwargs)

    def __call__(self, item):
        self.pbar.update(1)
        return item
