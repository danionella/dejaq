import numpy as np
import multiprocessing as mp

class ArrayFIFO:
    """
    A FIFO buffer (queue) for numpy arrays. The queue is implemented as a ring buffer in shared memory.
    """

    def __init__(self, bytes):
        """
        Initializes a ArrayFIFO object.

        Args:
            buffer_size (int): The size of the buffer in bytes.
        """

        self.buffer_size = int(bytes)
        self.buffer = mp.Array("c", self.buffer_size, lock=False)
        self._view = None
        self.queue = mp.Manager().Queue()  # manager helps avoid out-of-order problems
        self.get_lock = mp.Lock()
        self.put_lock = mp.Lock()
        self.head_changed = mp.Condition()
        self.head = mp.Value("i", 0)
        self.tail = mp.Value("i", 0)

    def put(self, array, meta=None):
        """
        Puts a byte array into the queue.

        Args:
            array (numpy.ndarray): The byte array to be put into the queue.
            meta (Any, optional): Additional metadata associated with the byte array.

        Raises:
            AssertionError: If the size of the byte array exceeds the buffer size.
        """

        array_bytes = array.ravel().view('byte')
        nbytes = array.nbytes

        assert nbytes < self.buffer_size, "Array size exceeds buffer size."

        with self.put_lock:
            while self._available_space() < nbytes:
                with self.head_changed:
                    self.head_changed.wait()
            tail = self.tail.value

            if tail + nbytes <= self.buffer_size:
                self.view[tail : tail + nbytes] = array_bytes
                self.tail.value = (tail + nbytes) % self.buffer_size
            else:
                tail_part_size = self.buffer_size - tail
                self.view[tail:] = array_bytes[:tail_part_size]
                self.view[: nbytes - tail_part_size] = array_bytes[tail_part_size:]
                self.tail.value = nbytes - tail_part_size

            array_info = dict(
                dtype=array.dtype.str, shape=array.shape, nbytes=nbytes, head=tail, tail=self.tail.value, meta=meta
            )
            self.queue.put(array_info)

    def get(self, callback=None, copy=None, **kwargs):
        """
        Gets a byte array from the queue.

        Args:
            callback (Callable, optional): A callback function to be called with the byte array (pre-copy, potentially unsafe!) and metadata.
            copy (bool, optional): Whether to make a copy of the byte array. Defaults to None: copy if a callback is not provided.
            **kwargs: Additional keyword arguments to be passed to the queue's get method.

        Returns:
            tuple: A tuple containing the byte array and any metadata provided with put.
        """
        with self.get_lock:
            array_info = self.queue.get(**kwargs)
            head = array_info["head"]
            tail = array_info["tail"]
            assert head == self.head.value, f"head: {head}, self.head: {self.head.value}"
            if head <= tail:
                array_bytes = self.view[head:tail]
            else:
                array_bytes = np.concatenate((self.view[head:], self.view[:tail]))
            array = np.frombuffer(array_bytes, dtype=array_info["dtype"]).reshape(array_info["shape"])
            if copy or ((copy is None) and (callback is None)):
                array = array.copy()
            if callback is not None:
                callback(array, array_info["meta"])
            self.head.value = (head + array_info["nbytes"]) % self.buffer_size

        with self.head_changed:
            self.head_changed.notify()

        return array, array_info["meta"]

    def _available_space(self):
        """
        Calculates the available space in the buffer.

        Returns:
            int: The available space in the buffer.
        """
        return (self.head.value - self.tail.value - 1) % self.buffer_size

    @property
    def view(self):
        """
        numpy.ndarray: A view of the shared memory array as a numpy array. Lazy initialization to avoid pickling issues.
        """
        if self._view is None:
            self._view = np.frombuffer(self.buffer, "byte")
        return self._view

    def __del__(self):
        self._view = None
