from __future__ import annotations
import os, sys, time, gc, weakref, struct
import multiprocessing as mp
from multiprocessing import shared_memory
import pickle
import dataclasses
from typing import Any
import numpy as np

IS_WIN = sys.platform.startswith("win")

class ByteFIFO:
    """ A FIFO buffer (queue) for bytes. The queue is implemented as a ring buffer in shared memory.
    """

    def __init__(self, buffer_bytes=10e6):
        """
        Initializes a ByteFIFO object.

        Args:
            buffer_bytes (int): The size of the buffer in bytes. Defaults to 10 MiB.
        """

        self.buffer_bytes = int(buffer_bytes)
        self.buffer = mp.Array("B", self.buffer_bytes, lock=False)
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
            #array_bytes = np.frombuffer(array_bytes, dtype='byte')
            array_bytes = array_bytes
        elif type(array_bytes) == np.ndarray:
            array_bytes = array_bytes.ravel().view('B')
        nbytes = array_bytes.nbytes
        assert nbytes < self.buffer_bytes, "Array size exceeds buffer size."
        with self.put_lock:
            while self._available_space() < nbytes:
                with self.head_changed:
                    if not self.head_changed.wait(timeout=timeout):
                        raise TimeoutError("Timeout waiting for available space.")
            _, frame_head, frame_tail = self._write_buffer(array_bytes)
            frame_info = FrameInfo(nbytes=nbytes, head=frame_head, tail=frame_tail, meta=meta)
            self.queue.put(frame_info)


    def _write_buffer(self, array_bytes, old_tail=None):
        ''' Write a byte array into the queue. Warning: this function should be called after acquiring the put_lock.
        '''
        old_tail = old_tail or self.tail.value
        nbytes = len(array_bytes)
        if old_tail + nbytes <= self.buffer_bytes:
            #print(type(array_bytes), array_bytes.nbytes, array_bytes.shape)
            self.view[old_tail : old_tail + nbytes] = array_bytes
            new_tail = (old_tail + nbytes) % self.buffer_bytes
        else:
            tail_part_size = self.buffer_bytes - old_tail
            self.view[old_tail:] = array_bytes[:tail_part_size]
            self.view[: nbytes - tail_part_size] = array_bytes[tail_part_size:]
            new_tail = nbytes - tail_part_size
        self.tail.value = new_tail
        return nbytes, old_tail, new_tail


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
            if frame_info is Ellipsis:
                self.close()
                return Ellipsis
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
            self.head.value = (head + frame_info.nbytes) % self.buffer_bytes

        with self.head_changed:
            self.head_changed.notify()

        return return_value

    def _available_space(self):
        """ Calculates the available space in the buffer.

        Returns:
            int: The available space in bytes.
        """
        return (self.head.value - self.tail.value - 1) % self.buffer_bytes

    @property
    def view(self):
        """ numpy.ndarray: A view of the shared memory array as a numpy array. Lazy initialization to avoid pickling issues.
        """
        if self._view is None:
            self._view = np.frombuffer(self.buffer, "B")
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

    def __iter__(self):
        while True:
            x = self.get()
            if x is Ellipsis: 
                self.queue.put(Ellipsis)
                return
            yield x
    
    def _signal_stop(self, n=1):
        ''' Puts n stop signals into the queue. '''
        for _ in range(n):
            self.queue.put(Ellipsis)

    def __getstate__(self):
        state = {k:v for k,v in self.__dict__.items() if k != '_view'}
        state['_view'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        

class ArrayFIFO(ByteFIFO):
    """ A fast queue for numpy arrays. 

    Args:
        buffer_bytes (int): The size of the buffer in bytes
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
    """ A fast queue for arbitrary (picklable) Python objects.

    Args:
        buffer_bytes (int): The size of the buffer in bytes. Defaults to 10 MiB.
    """
    def __init__(self, buffer_bytes=10e6):
        super().__init__(buffer_bytes=buffer_bytes)

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

        assert nbytes_total < self.buffer_bytes, "Array size exceeds buffer size."

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
    
    
@dataclasses.dataclass
class FrameInfo:
    ''' A class to store metadata about a data frame in a ring buffer.'''
    nbytes: int
    head: int
    tail: int
    meta: Any  # any picklable object




def _safe_base(prefix: str = "ns") -> str:
    return f"{prefix}{os.getpid():x}{os.urandom(2).hex()}"

def _posix_name(base: str) -> str:
    """Return a POSIX-safe name (macOS ≤31 bytes including '/')."""
    nm = "/" + "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    if sys.platform == "darwin":
        b = nm.encode()[:31]
        nm = b.decode("ascii", "ignore")
        if not nm.startswith("/"): nm = "/" + nm.lstrip("/")
    return nm

def _win_name(name: str) -> str:
    return name if name.startswith(("Local\\","Global\\")) else "Local\\" + name

class NamedSemaphore:
    """Cross-process named counting semaphore (picklable, best-effort cleanup)."""
    def __init__(self, name: str | None = None, create: bool = True,
                 initial: int = 0, maxcount: int | None = None, *, auto_unlink: bool = False) -> None:
        self.backend = "win32" if IS_WIN else "posix"
        if IS_WIN:
            import win32event, win32con
            MAX = int(maxcount if maxcount is not None else 2_147_483_647)
            if create:
                nm = _win_name(name or _safe_base("ns"))
                h = win32event.CreateSemaphore(None, int(initial), MAX, nm)
            else:
                if not name: raise ValueError("NamedSemaphore: name must be provided when create=False")
                nm = _win_name(name)
                h = win32event.OpenSemaphore(win32con.SEMAPHORE_MODIFY_STATE | win32con.SYNCHRONIZE,False, nm)
            if not h: raise OSError("Create/OpenSemaphore failed")
            self._h = h
            self.name = nm
        else:
            self.name = _posix_name(name.lstrip("/")) if name else _posix_name(_safe_base("ns"))
            import posix_ipc as P
            flags = P.O_CREAT | (P.O_EXCL if create else 0)
            if create:
                try:
                    self._sem = P.Semaphore(self.name, flags=flags, initial_value=int(initial))
                except Exception:
                    self._sem = P.Semaphore(self.name)
            else:
                self._sem = P.Semaphore(self.name)
        self._auto_unlink = bool(auto_unlink)
        weakref.finalize(self, NamedSemaphore._finalize, self.backend, self.name, self._auto_unlink)

    def acquire(self, timeout: float | None = None) -> bool:
        if IS_WIN:
            import win32event
            ms = win32event.INFINITE if timeout is None else max(0, int(timeout * 1000))
            rc = win32event.WaitForSingleObject(self._h, ms)
            if rc == win32event.WAIT_OBJECT_0:
                return True
            if rc == win32event.WAIT_TIMEOUT:
                return False
            raise RuntimeError(f"WaitForSingleObject rc={rc}")
        else: 
            import posix_ipc as P
            try:
                self._sem.acquire(timeout=None if timeout is None else float(timeout)); return True
            except P.BusyError:
                return False

    def release(self, n: int = 1) -> None:
        if IS_WIN:
            import win32event, pywintypes
            try:
                win32event.ReleaseSemaphore(self._h, int(n))
            except pywintypes.error as e:
                if getattr(e, "winerror", None) == 298:  # ERROR_TOO_MANY_POSTS
                    raise RuntimeError("Over-release of NamedSemaphore") from e
                raise
        else:
            for _ in range(int(n)): self._sem.release()

    def close(self) -> None:
        if IS_WIN:
            try: __import__("win32event").CloseHandle(self._h)
            except Exception: pass
        else:
            try: self._sem.close()
            except Exception: pass

    def unlink(self) -> None:
        if not IS_WIN:
            import posix_ipc as P
            try: P.unlink_semaphore(self.name)
            except Exception: pass

    def __getstate__(self) -> dict: return {"name": self.name, "backend": self.backend}
    def __setstate__(self, s: dict) -> None:
        self.__dict__.clear(); self.backend = s["backend"]
        self.__init__(s["name"], create=False)

    @staticmethod
    def _finalize(backend: str, name: str, auto_unlink: bool) -> None:
        if backend == "posix" and auto_unlink:
            try:
                import posix_ipc as P; P.unlink_semaphore(name)
            except Exception: pass

    def __del__(self):
        try: self.close()
        except Exception: pass

    def __enter__(self): self.acquire(); return self
    def __exit__(self, *_): self.release()

class NamedLock(NamedSemaphore):
    """Mutex from NamedSemaphore (maxcount=1). Detects over-release on POSIX."""
    def __init__(self, name: str | None = None, create: bool = True, *, auto_unlink: bool = False) -> None:
        super().__init__(name=name, create=create, initial=1, maxcount=1, auto_unlink=auto_unlink)

    def release(self) -> None:
        if IS_WIN:
            return super().release(1)
        # POSIX: probe to avoid silent over-release
        import posix_ipc as P
        try:
            self._sem.acquire(timeout=0)  # trywait
        except P.BusyError:
            self._sem.release()           # normal unlock
        else:
            self._sem.release()           # restore
            raise RuntimeError("Over-release of NamedLock on POSIX")

class NamedByteRing:
    """Manager/Condition-free ring buffer queue (bytes) with named semaphores.
       Pickleable across independently started processes.

    Args:
        buffer_bytes (int): Size of the ring buffer in bytes.
        name (str | None): Base name for shared memory and semaphores. If None (default), a random name is generated.
        create (bool): If True (default), attempt to create new shared memory and semaphores; if they already exist, open them. If False, only open existing resources.
        auto_unlink (bool): If True, automatically unlink shared memory and semaphores when the last reference is gone. Default True.
    """
    def __init__(self, buffer_bytes: int = 10_000_000, name: str | None = None, create: bool = True, *, auto_unlink: bool = True) -> None:
        base = name or _safe_base("nq")
        self.base = base
        self._auto_unlink = bool(auto_unlink)

        # Shared state: head, tail, capacity, closed (0/1)
        st_name = ("NS_"+base) if IS_WIN else base+"_S"
        self._owns_state = False
        if create:
            try:
                self.state = shared_memory.ShareableList([0, 0, int(buffer_bytes), 0], name=st_name)
                self._owns_state = True
            except FileExistsError:
                self.state = shared_memory.ShareableList(name=st_name)
        else:
            self.state = shared_memory.ShareableList(name=st_name)
        self._state_name = st_name

        # Data buffer
        buf_name = ("NB_"+base) if IS_WIN else base+"_B"
        total = int(buffer_bytes)
        self.cap = int(self.state[2])
        self._owns_buf = False
        if create:
            try:
                self.buf = shared_memory.SharedMemory(name=buf_name, create=True, size=total)
                self._owns_buf = True
                _view = np.frombuffer(self.buf.buf, dtype='B', count=total)
                _view[:] = 0
            except FileExistsError:
                self.buf = shared_memory.SharedMemory(name=buf_name, create=False)

        else:
            self.buf = shared_memory.SharedMemory(name=buf_name, create=False)
        self._buf_name = buf_name

        # Sync: serialize producers/consumers + count items + wake producers
        self.put_lock   = NamedLock(("NLp_"+base) if IS_WIN else base+"_Lp", create=create, auto_unlink=auto_unlink)
        self.get_lock   = NamedLock(("NLg_"+base) if IS_WIN else base+"_Lg", create=create, auto_unlink=auto_unlink)
        self.items      = NamedSemaphore(("NI_"+base)  if IS_WIN else base+"_I", create=create, initial=0, auto_unlink=auto_unlink)
        self.space_gate = NamedSemaphore(("NG_"+base)  if IS_WIN else base+"_G", create=create, initial=0, auto_unlink=auto_unlink)

        weakref.finalize(self, NamedByteRing._finalize, self._state_name, self._buf_name,
                         self._owns_state, self._owns_buf, self._auto_unlink)

    @property
    def closed(self) -> bool: return bool(int(self.state[3]))

    def _avail_space(self, head: int | None = None, tail: int | None = None) -> int:
        if head is None: head = int(self.state[0])
        if tail is None: tail = int(self.state[1])
        return (head - tail - 1) % self.cap

    def _write_bytes(self, data: memoryview | bytes, tail: int | None = None) -> int:
        """Write data at current tail; return new tail (mod cap). Caller holds put_lock."""
        if not isinstance(data, memoryview):
            data = memoryview(data)
        n = len(data); cap = self.cap
        tail = int(self.state[1]) if tail is None else tail
        end = tail + n
        if end <= cap:
            self.buf.buf[tail:end] = data
            new_tail = end % cap
        else:
            first = cap - tail
            self.buf.buf[tail:] = data[:first]
            self.buf.buf[:n-first] = data[first:]
            new_tail = n - first
        self.state[1] = int(new_tail)
        return int(new_tail)

    def _read_bytes(self, n: int, head: int | None = None) -> bytes:
        """Read n bytes from current head; advance head. Caller holds get_lock."""
        cap = self.cap
        head = int(self.state[0]) if head is None else head
        end = head + n
        if end <= cap:
            out = bytes(self.buf.buf[head:end])
            new_head = end % cap
        else:
            first = cap - head
            out = bytes(self.buf.buf[head:]) + bytes(self.buf.buf[:n-first])
            new_head = n - first
        self.state[0] = int(new_head)
        return out

    def put_bytes(self, payload: bytes, timeout: float | None = None) -> bool:
        """Enqueue an already-serialized message (bytes)."""
        total = 4 + len(payload)  # 4-byte length prefix
        deadline = None if timeout is None else (time.time() + float(timeout))
        while True:
            with self.put_lock:
                if self._avail_space() >= total:
                    # write len prefix then payload, then publish an item
                    self._write_bytes(struct.pack("<I", len(payload)))
                    self._write_bytes(payload)
                    ok = True
                else:
                    ok = False
            if ok:
                self.items.release(1)
                return True
            # wait for space
            if deadline is None:
                self.space_gate.acquire()  # wake when any consumer ran
            else:
                rem = deadline - time.time()
                if rem <= 0: return False
                if not self.space_gate.acquire(timeout=rem): return False

    def get_bytes(self, timeout: float | None = None) -> tuple[bool, bytes | None]:
        """Dequeue one message (bytes)."""
        if not self.items.acquire(timeout=timeout):
            return False, None
        with self.get_lock:
            (n,) = struct.unpack("<I", self._read_bytes(4))
            data = self._read_bytes(n)
        # signal one producer that space changed
        self.space_gate.release(1)
        return True, data

    def close(self) -> None:
        try: self.state.shm.close()
        except Exception: pass
        try: self.buf.close()
        except Exception: pass
        self.put_lock.close(); self.get_lock.close()
        self.items.close(); self.space_gate.close()

    def unlink(self) -> None:
        try: self.state.shm.unlink()
        except Exception: pass
        try: self.buf.unlink()
        except Exception: pass
        self.put_lock.unlink(); self.get_lock.unlink()
        self.items.unlink(); self.space_gate.unlink()

    def __getstate__(self) -> dict:
        return {
            "base": self.base,
            "state_name": self._state_name,
            "buf_name": self._buf_name,
            "locks": (self.put_lock.__getstate__(), self.get_lock.__getstate__()),
            "items": self.items.__getstate__(),
            "space_gate": self.space_gate.__getstate__(),
            "auto_unlink": self._auto_unlink,
            "cap": self.cap,
        }

    def __setstate__(self, s: dict) -> None:
        self.__dict__.clear()
        self.base = s["base"]; self._state_name = s["state_name"]; self._buf_name = s["buf_name"]; self.cap = s["cap"]
        self.state = shared_memory.ShareableList(name=self._state_name)
        self.buf = shared_memory.SharedMemory(name=self._buf_name)
        self.put_lock = NamedLock(s["locks"][0]["name"], create=False)
        self.get_lock = NamedLock(s["locks"][1]["name"], create=False)
        self.items = NamedSemaphore(s["items"]["name"], create=False)
        self.space_gate = NamedSemaphore(s["space_gate"]["name"], create=False)
        self._owns_state = False; self._owns_buf = False
        self._auto_unlink = bool(s.get("auto_unlink", False))
        weakref.finalize(self, NamedByteRing._finalize, self._state_name, self._buf_name,
                         self._owns_state, self._owns_buf, self._auto_unlink)

    @staticmethod
    def _finalize(state_name: str, buf_name: str,
                  owns_state: bool, owns_buf: bool, auto_unlink: bool) -> None:
        gc.collect()
        if not auto_unlink: return
        try:
            if owns_state:
                shm = shared_memory.SharedMemory(name=state_name); shm.unlink(); shm.close()
        except Exception: pass
        try:
            if owns_buf:
                shm = shared_memory.SharedMemory(name=buf_name); shm.unlink(); shm.close()
        except Exception: pass

class PicklableDejaQueue(NamedByteRing):
    """Pickleable queue for arbitrary Python objects"""

    def put(self, obj, timeout: float | None = None) -> bool:
        bufs = []
        p0 = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=bufs.append)
        segs = [memoryview(p0)] + [b.raw() for b in bufs]
        K = len(segs)
        lens = [len(s) for s in segs]
        hdr = struct.pack("<I", K) + struct.pack("<" + "I"*K, *lens)
        need = len(hdr) + sum(lens)

        deadline = None if timeout is None else (time.time() + float(timeout))
        while True:
            with self.put_lock:
                if self._avail_space() >= need:
                    self._write_bytes(hdr)
                    for s in segs: self._write_bytes(s)
                    ok = True
                else:
                    ok = False
            if ok:
                self.items.release(1); return True
            if deadline is None:
                self.space_gate.acquire()
            else:
                rem = deadline - time.time()
                if rem <= 0 or not self.space_gate.acquire(timeout=rem): return False

    def get(self, timeout: float | None = None):
        if not self.items.acquire(timeout=timeout): return None

        with self.get_lock:
            cap = self.cap
            head0 = int(self.state[0])
            buf = self.buf.buf

            def _copy_span(start, n):
                start %= cap; end = start + n
                if end <= cap: return bytes(buf[start:end])
                first = cap - start
                return bytes(buf[start:]) + bytes(buf[:n-first])

            # header (copy — small)
            K = struct.unpack("<I", _copy_span(head0, 4))[0]
            lens = struct.unpack("<" + "I"*K, _copy_span(head0 + 4, 4*K))

            total = 4 + 4*K + sum(lens)
            # segments: zero-copy if contiguous, else copy
            segs, cursor = [], head0 + 4 + 4*K
            for n in lens:
                s = cursor % cap; e = s + n
                if e <= cap:
                    segs.append(memoryview(buf)[s:e])
                else:
                    segs.append(_copy_span(cursor, n))  # wrapped -> copy
                cursor += n

            obj = pickle.loads(segs[0], buffers=[m for m in segs[1:]])
            self.state[0] = (head0 + total) % cap  # advance after loads()

        self.space_gate.release(1)
        return obj
    
    def __iter__(self):
        while True:
            x = self.get()
            if x is Ellipsis: 
                self.put(Ellipsis)
                return
            yield x
    
    def _signal_stop(self, n=1):
        ''' Puts n stop signals into the queue. '''
        for _ in range(n):
            self.put(Ellipsis)



