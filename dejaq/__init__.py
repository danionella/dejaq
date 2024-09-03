import numpy as np
import multiprocessing as mp
import pickle
import dataclasses
from typing import Any

@dataclasses.dataclass
class FrameInfo:
    ''' A class to store metadata about a data frame in a FIFO queue.'''
    nbytes: int
    head: int
    tail: int
    meta: Any  # any picklable object

class ByteFIFO:
    """ A FIFO buffer (queue) for bytes. The queue is implemented as a ring buffer in shared memory.
    """

    def __init__(self, bytes=10e6):
        """
        Initializes a ByteFIFO object.

        Args:
            buffer_size (int): The size of the buffer in bytes. Defaults to 10 MiB.
        """

        self.buffer_size = int(bytes)
        self.buffer = mp.Array("c", self.buffer_size, lock=False)
        self._view = None
        self.queue = mp.Manager().Queue()  # manager helps avoid out-of-order problems
        self.get_lock = mp.Lock()
        self.put_lock = mp.Lock()
        self.head_changed = mp.Condition()
        self.head = mp.Value("l", 0)
        self.tail = mp.Value("l", 0)
        self.closed = mp.Value("b", False)


    def put(self, array_bytes, meta=None, timeout=None):
        """
        Puts a byte array into the queue.

        Args:
            array_bytes (numpy.ndarray or memoryview): The byte array to be put into the queue.
            meta (Any, optional): Additional metadata associated with the byte array.

        Raises:
            AssertionError: If the size of the byte array exceeds the buffer size.
        """
        if type(array_bytes) == memoryview:
            array_bytes = np.frombuffer(array_bytes, dtype='byte')
        elif type(array_bytes) == np.ndarray:
            array_bytes = array_bytes.ravel().view('byte')
        nbytes = array_bytes.nbytes
        assert nbytes < self.buffer_size, "Array size exceeds buffer size."
        with self.put_lock:
            while self._available_space() < nbytes:
                with self.head_changed:
                    if not self.head_changed.wait(timeout=timeout):
                        raise TimeoutError("Timeout waiting for available space.")
            head = self.tail.value
            self._write_buffer(array_bytes)
            frame_info = FrameInfo(nbytes=nbytes, head=head, tail=self.tail.value, meta=meta)
            self.queue.put(frame_info)


    def _write_buffer(self, array_bytes):
        ''' Write a byte array into the queue. Warning: this function should be called after acquiring the put_lock.
        '''
        tail = self.tail.value
        nbytes = len(array_bytes)
        if tail + nbytes <= self.buffer_size:
            self.view[tail : tail + nbytes] = array_bytes
            self.tail.value = (tail + nbytes) % self.buffer_size
        else:
            tail_part_size = self.buffer_size - tail
            self.view[tail:] = array_bytes[:tail_part_size]
            self.view[: nbytes - tail_part_size] = array_bytes[tail_part_size:]
            self.tail.value = nbytes - tail_part_size
        return nbytes


    def get(self, callback=None, copy=None, **kwargs):
        """ Gets a byte array from the queue.

        Args:
            callback (Callable, optional): A callback function to be called with the byte array (pre-copy, potentially unsafe!) and metadata.
            copy (bool, optional): Whether to make a copy of the byte array. Defaults to None: copy if a callback is not provided.
            **kwargs: Additional keyword arguments to be passed to the queue's get method.

        Returns:
            A tuple containing the byte array and any metadata provided with put OR the return value of the callback function, if provided.
        """
        with self.get_lock:
            frame_info = self.queue.get(**kwargs)
            head = frame_info.head
            tail = frame_info.tail
            assert head == self.head.value, f"head: {head}, self.head: {self.head.value}"
            if head <= tail:
                array_bytes = self.view[head:tail]
            else:
                array_bytes = np.concatenate((self.view[head:], self.view[:tail]))
            if copy or ((copy is None) and (callback is None)):
                array_bytes = array_bytes.copy()
            if callback is not None:
                return_value = callback(array_bytes, frame_info.meta)
            else:
                return_value = array_bytes, frame_info.meta
            self.head.value = (head + frame_info.nbytes) % self.buffer_size

        with self.head_changed:
            self.head_changed.notify()

        return return_value

    def _available_space(self):
        """ Calculates the available space in the buffer.

        Returns:
            int: The available space in bytes.
        """
        return (self.head.value - self.tail.value - 1) % self.buffer_size

    @property
    def view(self):
        """ numpy.ndarray: A view of the shared memory array as a numpy array. Lazy initialization to avoid pickling issues.
        """
        if self._view is None:
            self._view = np.frombuffer(self.buffer, "byte")
        return self._view

    def __del__(self):
        self._view = None

    def empty(self):
        """ Checks if the queue is empty.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return self.queue.empty()

    def close(self):
        """ Closes the queue.
        """
        self.closed.value = True

    def join(self):
        """ Joins the queue
        """
        self.queue.join()

    def task_done(self):
        """ Marks a task as done.
        """
        self.queue.task_done()

    @property
    def done(self):
        ''' Returns True if the queue is empty and closed.'''
        return self.queue.empty() and self.closed.value

    def __getstate__(self):
        state = {k:v for k,v in self.__dict__.items() if k != '_view'}
        state['_view'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        

class ArrayFIFO(ByteFIFO):
    """ A fast queue for numpy arrays. The queue is implemented as a ring buffer in shared memory.

    Args:
        buffer_size (int): The size of the buffer in bytes
    """

    def put(self, array, meta=None, timeout=None):
        """
        Puts a numpy array into the queue.

        Args:
            array (numpy.ndarray): The byte array to be put into the queue.
            meta (Any, optional): Additional custom metadata (will be sent through a regular slow Queue).
            timeout (float, optional): The maximum time to wait for available space in the queue.

        Raises:
            AssertionError: If the size of the byte array exceeds the buffer size.
        """

        array_bytes = array.ravel().view('byte')
        meta = dict(dtype=array.dtype.str, shape=array.shape, meta=meta)
        super().put(array_bytes, meta=meta, timeout=timeout)

    def get(self, callback=None, copy=None, **kwargs):
        """
        Gets a numpy array from the queue.

        Args:
            callback (Callable, optional): A callback function to be called with the byte array (pre-copy, potentially unsafe!) and metadata.
            copy (bool, optional): Whether to make a copy of the byte array. Defaults to None: copy if a callback is not provided.
            **kwargs: Additional keyword arguments to be passed to the queue's get method.

        Returns:
            tuple: A tuple containing the numpy array and any metadata provided with put.
        """
        def callback_wrapper(array_bytes, meta):
            array = np.frombuffer(array_bytes, dtype=meta['dtype']).reshape(meta['shape'])
            if copy or ((copy is None) and (callback is None)):
                array = array.copy()
            if callback is not None:
                callback(array, meta)
            return array, meta['meta']

        array, meta = super().get(copy=False, callback=callback_wrapper, **kwargs)
        return array, meta

class DejaQueue(ByteFIFO):
    """ A fast queue for arbitrary (picklable) Python objects. The queue is implemented as a ring buffer in shared memory.

    Args:
        buffer_size (int): The size of the buffer in bytes.
    """

    def put(self, obj, timeout=None):
        """ Puts a Python object into the queue.

        Args:
            obj (Any): The byte array to be put into the queue.
            timeout (float, optional): The maximum time to wait for available space in the queue.
        """
        buffers = []
        pkl = pickle.dumps(obj, buffer_callback=buffers.append, protocol=pickle.HIGHEST_PROTOCOL)
        buffer_lengths = [len(pkl),] + [len(it.raw()) for it in buffers]
        nbytes_total = sum(buffer_lengths)

        assert nbytes_total < self.buffer_size, "Array size exceeds buffer size."

        with self.put_lock:
            while self._available_space() < nbytes_total:
                with self.head_changed:
                    if not self.head_changed.wait(timeout=timeout):
                        raise TimeoutError("Timeout waiting for available space.")

            head = self.tail.value
            self._write_buffer(np.frombuffer(pkl, 'byte'))
            for buf in buffers:
                self._write_buffer(buf.raw())

            frame_info = FrameInfo(nbytes=nbytes_total, head=head, tail=self.tail.value, meta=buffer_lengths)
            self.queue.put(frame_info)

    def get(self, **kwargs):
        """ Gets an item from the queue.

        Args:
            **kwargs: Additional keyword arguments to be passed to the underlying queue's get method (e.g. timeout).

        Returns:
            obj: The object that was put into the queue.
        """
        def callback(array_bytes, buffer_lengths):
            buffers = []
            offset = 0
            for length in buffer_lengths:
                buffers.append(pickle.PickleBuffer(array_bytes[offset:offset+length]))
                offset += length
            obj = pickle.loads(buffers[0], buffers=buffers[1:])
            return obj

        obj = super().get(copy=False, callback=callback, **kwargs)
        return obj
        


class lazymap:
    ''' Returns an Iterable, functionally related to map (except that outputs are calculated by a pool of processes).
    
    Args:
        fcn (callable): function that is being mapped. Signature: fcn(item, **kwargs)
        it (iterable): iterable that maps over the function, providing items as arguments
        num_workers (int): number of workers (default: 1)
        buffer_size (int): size of the queue buffer (default: 10e6 bytes)
        **kwargs: optional, being passed to fcn

    Returns: 
        (iterable): an iterable that returns the results of fcn(item) for each item in it
    '''

    def __init__(self, fcn, it, num_workers=1, buffer_size=10e6, **kwargs):
        self.it = it
        self.in_queue = DejaQueue(buffer_size)
        self.out_queue = DejaQueue(buffer_size)
        self.k = mp.Value("l", 0)
        self.k_changed = mp.Condition()
        self.num_workers = num_workers
        self.workers = [mp.Process(target=self._worker_fcn, args=(pid, fcn), kwargs=kwargs) for pid in range(num_workers)]
        [w.start() for w in self.workers]
        self.generator = self.lazymap_generator(it, num_workers)

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
                self.k_changed.wait_for(lambda: self.k.value % self.num_workers == pid)
                self.out_queue.put(res)
                self.k.value += 1
                self.k_changed.notify_all()

    def lazymap_generator(self, it, num_workers):
        it = iter(it)
        [self.in_queue.put(next(it)) for i in range(num_workers)]
        k = -1
        for k, item in enumerate(it):
            res = self.out_queue.get()
            self.in_queue.put(item)
            yield res
        for k in range(k + 1, k + 1 + num_workers):
            yield self.out_queue.get()
        self.close()

    def close(self):
        [self.in_queue.put(None) for w in self.workers]
        [w.join() for w in self.workers]
        self.out_queue = None
        self.in_queue = None