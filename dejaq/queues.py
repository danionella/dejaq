from __future__ import annotations
import logging
import errno, os, sys, time, gc, weakref, struct, threading
import multiprocessing as mp
from multiprocessing import shared_memory
from copy import deepcopy
import pickle
import cloudpickle
import dataclasses
from typing import Any
import numpy as np
import array

logger = logging.getLogger(__name__)

class _pickleall:
    """Pickle backend that tries pickle first, then cloudpickle."""
    def dumps(obj, protocol=None, *, buffer_callback=None):
        # Out-of-band buffers are collected first and only handed on from the attempt that succeeded:
        # pickle may emit some buffers before it fails, which would otherwise mix with cloudpickle's.
        bufs = []
        try:
            data = pickle.dumps(obj, protocol=protocol, buffer_callback=bufs.append)
        except (pickle.PicklingError, AttributeError, TypeError):
            # lambdas and local functions/classes: PicklingError on 3.14+, AttributeError before
            bufs = []
            data = cloudpickle.dumps(obj, protocol=protocol, buffer_callback=bufs.append)
        if buffer_callback is not None:
            for b in bufs:
                buffer_callback(b)
        return data

    def loads(*args, **kwargs):
        return pickle.loads(*args, **kwargs)
    
    def PickleBuffer(*args, **kwargs):
        return pickle.PickleBuffer(*args, **kwargs)

_pickle_backend = _pickleall

_IS_WIN = sys.platform.startswith("win")
_PICKLE_PROTOCOL = 5


def _safe_base(prefix: str = "ns") -> str:
    """Random base name: prefix + pid (to tell leftover names apart) + 40 random bits.

    At most 2 + 8 + 10 characters, so semaphore and shared-memory names derived from it stay within the
    macOS name limits without shortening. Creators still retry on the (now rare) collision.
    """
    return f"{prefix}{os.getpid():x}{os.urandom(5).hex()}"


_CREATE_ATTEMPTS = 16  # fresh random names tried before giving up on a name collision


# macOS limits on POSIX names, in bytes including the leading '/' (measured on macOS CI:
# posix_ipc rejects 31-byte semaphore names; shm_open rejects 32-byte names)
_DARWIN_SEM_NAME_MAX = 30
_DARWIN_SHM_NAME_MAX = 31


def _darwin_shorten(nm: str, max_len: int) -> str:
    """On macOS, shorten a too-long '/'-prefixed name to a prefix plus a hash of the full name.

    Different long names (e.g. a queue's '_Lp' and '_Lg' locks) stay distinct, every process
    derives the same short name, and the result is a fixed point (shortening it again is a no-op).
    Names that fit, and all names on other platforms, are returned unchanged, so processes running
    different dejaq versions still agree on them.
    """
    if sys.platform != "darwin" or len(nm.encode()) <= max_len:
        return nm
    import hashlib

    digest = hashlib.blake2s(nm.encode(), digest_size=6).hexdigest()  # 12 hex chars
    keep = max_len - len(digest) - 1
    return nm.encode()[:keep].decode("utf-8", "ignore") + "-" + digest


def _posix_name(base: str) -> str:
    """Return a POSIX-safe semaphore name ('/'-prefixed; short enough for macOS)."""
    nm = "/" + "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    return _darwin_shorten(nm, _DARWIN_SEM_NAME_MAX)


def _shm_name(name: str) -> str:
    """Shared-memory name for SharedMemory (which adds the '/'), short enough for macOS."""
    return _darwin_shorten("/" + name, _DARWIN_SHM_NAME_MAX)[1:]


def _win_name(name: str) -> str:
    return name if name.startswith(("Local\\", "Global\\")) else "Local\\" + name


_LONG_MAX = 2_147_483_647  # semaphore counts are 32-bit LONGs on Windows

if _IS_WIN:
    # Kernel32 semaphore API without pywin32 (which has no free-threaded build). Waiting and closing
    # use the stdlib's _winapi (C, releases the GIL, a bit faster than ctypes); creating, opening and
    # releasing semaphores, which _winapi lacks, use ctypes. A private WinDLL instance (not
    # ctypes.windll.kernel32) keeps these argtypes from clashing with other libraries.
    import _winapi
    import ctypes
    from ctypes import wintypes as _W

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _CreateSemaphoreW = _kernel32.CreateSemaphoreW
    _CreateSemaphoreW.argtypes = [ctypes.c_void_p, _W.LONG, _W.LONG, _W.LPCWSTR]
    _CreateSemaphoreW.restype = _W.HANDLE
    _OpenSemaphoreW = _kernel32.OpenSemaphoreW
    _OpenSemaphoreW.argtypes = [_W.DWORD, _W.BOOL, _W.LPCWSTR]
    _OpenSemaphoreW.restype = _W.HANDLE
    _ReleaseSemaphore = _kernel32.ReleaseSemaphore
    _ReleaseSemaphore.argtypes = [_W.HANDLE, _W.LONG, ctypes.POINTER(_W.LONG)]
    _ReleaseSemaphore.restype = _W.BOOL
    _WaitForSingleObject = _winapi.WaitForSingleObject  # raises OSError on WAIT_FAILED
    _CloseHandle = _winapi.CloseHandle

    _INFINITE = _winapi.INFINITE
    _WAIT_OBJECT_0 = _winapi.WAIT_OBJECT_0
    _WAIT_TIMEOUT = _winapi.WAIT_TIMEOUT
    _SEMAPHORE_MODIFY_STATE = 0x0002
    _SYNCHRONIZE = 0x00100000
    _ERROR_TOO_MANY_POSTS = 298
    _ERROR_ALREADY_EXISTS = 183

    class _WinHandle:
        """Owns a kernel handle; close() closes it once and zeroes it, so a closed handle is never
        passed to the OS again (Windows reuses handle values, so a stale value may name another object)."""

        __slots__ = ("value",)

        def __init__(self, value: int) -> None:
            self.value = value

        def close(self, _close=_CloseHandle) -> None:  # bound at definition: safe during shutdown
            h, self.value = self.value, 0
            if h:
                _close(h)

        def get(self) -> int:
            h = self.value
            if not h:
                raise ValueError("operation on closed NamedSemaphore")
            return h


# On POSIX, SharedMemory(name=...) registers the segment with the calling process's
# resource_tracker even when it only attaches to an existing segment. A process started
# independently (e.g. via subprocess, not as a multiprocessing child) has its own tracker,
# which unlinks every registered segment when that process exits -- destroying the
# creator's queue for everyone else. Attachers must therefore not register; the creator
# keeps its registration (and with it crash cleanup).
_untracked = threading.local()
_register_patch_lock = threading.Lock()
_register_patched = False


def _patch_register_once() -> None:
    """Python < 3.13 (no ``track`` argument): let ``_attach_shm`` suppress registration.

    Unregistering right after attaching is not an option: a multiprocessing child shares its
    parent's tracker, so it would remove the creator's registration (and the creator's own
    ``unlink()`` would later make the tracker raise KeyError). Instead, wrap
    ``resource_tracker.register`` so that it is skipped only in the current thread while
    ``_attach_shm`` runs; every other call passes through unchanged.
    """
    global _register_patched
    with _register_patch_lock:
        if _register_patched:
            return
        from multiprocessing import resource_tracker

        orig = resource_tracker.register

        def register(name, rtype):
            if rtype == "shared_memory" and getattr(_untracked, "active", False):
                return
            return orig(name, rtype)

        resource_tracker.register = register
        _register_patched = True


def _attach_shm(name: str, cls=shared_memory.SharedMemory) -> shared_memory.SharedMemory:
    """Attach to existing shared memory without registering it with this process's resource tracker."""
    if _IS_WIN:
        return cls(name=name)
    if sys.version_info >= (3, 13):
        return cls(name=name, track=False)
    _patch_register_once()
    _untracked.active = True
    try:
        return cls(name=name)
    finally:
        _untracked.active = False


class NamedSemaphore:
    """Cross-process named counting semaphore (picklable, best-effort cleanup).

    Args:
        name: Name of the semaphore. If None (only with create=True), a random unique name is generated.
        create: If True (default), create a new semaphore; raises FileExistsError if ``name`` already exists.
            If False, open an existing one.
        initial: Initial count (create=True only).
        maxcount: Maximum count (Windows only; default: the largest allowed).
        auto_unlink: If True (default), the creating process unlinks the name once the object is gone (POSIX).
    """

    def __init__(
        self,
        name: str | None = None,
        create: bool = True,
        initial: int = 0,
        maxcount: int | None = None,
        *,
        auto_unlink: bool = True,
    ) -> None:
        self.backend = "win32" if _IS_WIN else "posix"
        if _IS_WIN:
            MAX = int(maxcount if maxcount is not None else _LONG_MAX)
            if create:
                if not (1 <= MAX <= _LONG_MAX and 0 <= int(initial) <= MAX):
                    raise ValueError(f"NamedSemaphore: need 0 <= initial <= maxcount <= {_LONG_MAX}")
                for _ in range(_CREATE_ATTEMPTS):
                    nm = _win_name(name or _safe_base("ns"))
                    ctypes.set_last_error(0)
                    # opens the existing semaphore (ignoring initial/max) if the name already exists
                    h = _CreateSemaphoreW(None, int(initial), MAX, nm)
                    if not h or ctypes.get_last_error() != _ERROR_ALREADY_EXISTS:
                        break
                    _CloseHandle(h)  # someone else's semaphore: never share it
                    h = None
                    if name:
                        break
                if h is None:
                    raise FileExistsError(errno.EEXIST, "NamedSemaphore already exists", nm)
            else:
                if not name:
                    raise ValueError("NamedSemaphore: name must be provided when create=False")
                nm = _win_name(name)
                h = _OpenSemaphoreW(_SEMAPHORE_MODIFY_STATE | _SYNCHRONIZE, False, nm)
            if not h:
                raise ctypes.WinError(ctypes.get_last_error())
            self._h = _WinHandle(h)
            self.name = nm
        else:
            import posix_ipc as P

            if create:
                for _ in range(_CREATE_ATTEMPTS):
                    nm = _posix_name(name.lstrip("/")) if name else _posix_name(_safe_base("ns"))
                    try:
                        self._sem = P.Semaphore(nm, flags=P.O_CREAT | P.O_EXCL, initial_value=int(initial))
                        break
                    except P.ExistentialError:  # name taken: never share someone else's semaphore
                        if name:
                            break
                else:
                    nm = None
                if getattr(self, "_sem", None) is None:
                    raise FileExistsError(errno.EEXIST, "NamedSemaphore already exists", nm)
                self.name = nm
            else:
                if not name:
                    raise ValueError("NamedSemaphore: name must be provided when create=False")
                self.name = _posix_name(name.lstrip("/"))
                self._sem = P.Semaphore(self.name)
        self._auto_unlink = bool(auto_unlink)
        handle = self._h if _IS_WIN else self._sem
        # only the creating process unlinks automatically (not e.g. a forked child that inherited the object)
        owner_pid = os.getpid() if create else None
        self._finalizer = weakref.finalize(
            self, NamedSemaphore._finalize, self.backend, self.name, self._auto_unlink, owner_pid, handle
        )

    def acquire(self, timeout: float | None = None) -> bool:
        if _IS_WIN:
            if timeout is None:
                ms = _INFINITE
            else:  # clamp to the largest finite DWORD timeout
                ms = min(max(0, int(timeout * 1000)), _INFINITE - 1)
            rc = _WaitForSingleObject(self._h.get(), ms)
            if rc == _WAIT_OBJECT_0:
                return True
            if rc == _WAIT_TIMEOUT:
                return False
            raise RuntimeError(f"WaitForSingleObject rc={rc}")
        else:
            import posix_ipc as P
            if timeout is None:
                self._sem.acquire()
                return True

            timeout = max(0.0, float(timeout))

            if P.SEMAPHORE_TIMEOUT_SUPPORTED:
                try:
                    self._sem.acquire(timeout=timeout)
                    return True
                except P.BusyError:
                    return False
            else: # polling hack for MACOS, which doesn't support timeout
                deadline = time.monotonic() + timeout
                sleep_s = 0.0005
                while True:
                    try: 
                        self._sem.acquire(timeout=0)  # trywait
                        return True
                    except P.BusyError:
                        if time.monotonic() >= deadline:
                            return False
                        time.sleep(sleep_s)
                        sleep_s = min(0.01, sleep_s * 2)

    def release(self, n: int = 1) -> None:
        if _IS_WIN:
            n = int(n)
            if not 1 <= n <= _LONG_MAX:  # ctypes would silently truncate out-of-range counts
                raise ValueError(f"NamedSemaphore.release: n must be in [1, {_LONG_MAX}], got {n}")
            if not _ReleaseSemaphore(self._h.get(), n, None):
                err = ctypes.WinError(ctypes.get_last_error())
                if err.winerror == _ERROR_TOO_MANY_POSTS:
                    raise RuntimeError("Over-release of NamedSemaphore") from err
                raise err
        else:
            for _ in range(int(n)):
                self._sem.release()

    def close(self) -> None:
        self._finalizer()

    def unlink(self) -> None:
        if not _IS_WIN:
            import posix_ipc as P
            try:
                P.unlink_semaphore(self.name)
            except Exception:
                pass

    def __getstate__(self) -> dict:
        return {"name": self.name, "backend": self.backend, "auto_unlink": self._auto_unlink}

    def __setstate__(self, s: dict) -> None:
        self.__dict__.clear()
        self.backend = s["backend"]
        auto_unlink = bool(s.get("auto_unlink", False))
        self.__init__(s["name"], create=False, auto_unlink=auto_unlink)

    @staticmethod
    def _finalize(backend: str, name: str, auto_unlink: bool, owner_pid: int | None, handle: Any) -> None:
        try:
            if backend == "win32":
                try:
                    logger.debug(f"finalizer closing {name}")
                    handle.close()
                except Exception:
                    pass
            else:
                try:
                    handle.close()
                    logger.debug(f"finalizer closing {name}")
                except Exception:
                    pass
                if auto_unlink and owner_pid == os.getpid():
                    try:
                        import posix_ipc as P
                        P.unlink_semaphore(name)
                        logger.debug(f"finalizer unlinking {name}")
                    except Exception:
                        pass
        except Exception:
            pass

    def __del__(self):
        finalizer = getattr(self, "_finalizer", None)  # absent if __init__ failed
        if finalizer is not None:
            finalizer()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()


class NamedLock(NamedSemaphore):
    """Mutex from NamedSemaphore (maxcount=1). Detects over-release on POSIX."""

    def __init__(self, name: str | None = None, create: bool = True, *, auto_unlink: bool = True) -> None:
        super().__init__(name=name, create=create, initial=1, maxcount=1, auto_unlink=auto_unlink)

    def release(self) -> None:
        if _IS_WIN:
            return super().release(1)
        # POSIX: probe to avoid silent over-release
        import posix_ipc as P

        try:
            self._sem.acquire(timeout=0)  # trywait
        except P.BusyError:
            self._sem.release()  # normal unlock
        else:
            self._sem.release()  # restore
            raise RuntimeError("Over-release of NamedLock on POSIX")


class _SharedMemory(shared_memory.SharedMemory):
    """SharedMemory that owns the views derived from it and releases them in close().

    A live derived view (e.g. ``buf.cast("q")``) keeps the mapping exported, so closing fails with
    ``BufferError: cannot close exported pointers exist``. Normally the view is dropped first, but the
    garbage collector may finalize the SharedMemory before its views (seen at interpreter shutdown on
    free-threaded CPython 3.14t). Releasing owned views in close() makes the order irrelevant.
    """

    _views: tuple = ()

    def cast(self, fmt: str) -> memoryview:
        view = self.buf.cast(fmt)
        self._views = (*self._views, view)
        return view

    def close(self) -> None:
        views, self._views = self._views, ()
        for view in views:
            view.release()
        super().close()


# Shared state block: int64 slots, read and written only under _state_lock.
#   _WAITERS counts producers registered as waiting for space (see DejaQueue.put).
#   _MAGIC identifies the layout of this block and of the items in the ring: processes sharing a queue
#   must run compatible dejaq versions, and a mismatch is reported instead of corrupting data.
_HEAD, _TAIL, _NITEMS, _CLOSED, _CAP, _WAITERS, _MAGIC = range(7)
_STATE_SLOTS = 7
_STATE_MAGIC = 0x64656A61710002  # "dejaq" + format version 2 (version 1: 5 slots, 32-bit item lengths)


def _pack_header(lens) -> bytes:
    """Item header: number of segments K, then K segment lengths (all uint64)."""
    return struct.pack(f"<Q{len(lens)}Q", len(lens), *lens)


class NamedByteRing:
    """Manager/Condition-free ring buffer queue (bytes) with named semaphores.
       Pickleable across independently started processes.

    Args:
        buffer_bytes (int): Size of the ring buffer in bytes.
        name (str | None): Base name for shared memory and semaphores. If None (default), a random unique name is generated.
        create (bool): If True (default), create new shared memory and semaphores; raises FileExistsError if a
            queue with this name exists. If False, open an existing queue.
        auto_unlink (bool): If True (default), the creating process unlinks the shared memory and semaphores once
            its queue object is gone. Other processes must not attach after that.
    """

    def __init__(
        self, buffer_bytes: int = 10_000_000, name: str | None = None, create: bool = True, *, auto_unlink: bool = True
    ) -> None:
        self._auto_unlink = bool(auto_unlink)
        self._owns = False  # set only once creation has succeeded: a failed __init__ must never unlink anything
        if not create:
            self._open(name or _safe_base("nq"))
        elif name is not None:
            self._create(name, int(buffer_bytes))
        else:
            for attempt in range(_CREATE_ATTEMPTS):
                try:
                    self._create(_safe_base("nq"), int(buffer_bytes))
                    break
                except FileExistsError:  # random name already taken: try another
                    if attempt == _CREATE_ATTEMPTS - 1:
                        raise
        self._register_finalizer()

    @staticmethod
    def _names(base: str):
        """Names of (state shm, buffer shm, [put lock, get lock, state lock, items, space gate]) for a base name."""
        if _IS_WIN:
            return "NS_" + base, "NB_" + base, [p + base for p in ("NLp_", "NLg_", "NLs_", "NI_", "NG_")]
        return _shm_name(base + "_S"), _shm_name(base + "_B"), [base + s for s in ("_Lp", "_Lg", "_Ls", "_I", "_G")]

    def _create_sem(self, cls, name: str):
        try:
            return cls(name, create=True, auto_unlink=self._auto_unlink)
        except FileExistsError:
            if _IS_WIN:
                raise
        # POSIX semaphores outlive the processes that created them, e.g. after a crash. The caller has just
        # created this queue's shared memory exclusively, so a semaphore under its name can only belong to a
        # dead queue: replace it. (Processes still attached to the dead queue keep their own semaphore.)
        import posix_ipc as P

        try:
            P.unlink_semaphore(_posix_name(name))
        except P.ExistentialError:
            pass
        return cls(name, create=True, auto_unlink=self._auto_unlink)

    def _create(self, base: str, buffer_bytes: int) -> None:
        st_name, buf_name, sem_names = self._names(base)
        created = []  # cleaned up again if any step fails
        try:
            state_mem = _SharedMemory(create=True, name=st_name, size=8 * _STATE_SLOTS)
            created.append(state_mem)
            buf = shared_memory.SharedMemory(name=buf_name, create=True, size=buffer_bytes)
            created.append(buf)
            _view = np.frombuffer(buf.buf, dtype="B", count=buffer_bytes)
            _view[:] = 0
            del _view
            sems = []
            for cls, nm in zip((NamedLock, NamedLock, NamedLock, NamedSemaphore, NamedSemaphore), sem_names):
                sems.append(self._create_sem(cls, nm))
                created.append(sems[-1])
            state = state_mem.cast("q")
            state[0:_STATE_SLOTS] = array.array("q", [0, 0, 0, 0, buffer_bytes, 0, _STATE_MAGIC])
        except BaseException:
            for obj in reversed(created):
                for step in (obj.unlink, obj.close):
                    try:
                        step()
                    except Exception:
                        pass
            raise
        self._base = base
        self._state_mem, self._state, self.buf, self.cap = state_mem, state, buf, buffer_bytes
        self._state_name, self._buf_name = st_name, buf_name
        self._put_lock, self._get_lock, self._state_lock, self._items, self._space_gate = sems
        self._owns = True
        self._creator_pid = os.getpid()

    def _open(self, base: str) -> None:
        st_name, buf_name, sem_names = self._names(base)
        self._base = base
        self._state_mem = _attach_shm(st_name, cls=_SharedMemory)
        state = self._state_mem.cast("q")
        if len(state) < _STATE_SLOTS or state[_MAGIC] != _STATE_MAGIC:
            raise RuntimeError(
                f"Queue {base!r} was created by an incompatible dejaq version; "
                "all processes sharing a queue must use the same dejaq version."
            )
        self._state = state
        self.cap = int(state[_CAP])  # read cap stored by the creator
        self.buf = _attach_shm(buf_name)
        self._state_name, self._buf_name = st_name, buf_name
        self._put_lock, self._get_lock, self._state_lock = (NamedLock(nm, create=False) for nm in sem_names[:3])
        self._items, self._space_gate = (NamedSemaphore(nm, create=False) for nm in sem_names[3:])

    def _register_finalizer(self) -> None:
        owner_pid = self._creator_pid if self._owns else None
        weakref.finalize(
            self, NamedByteRing._finalize, self._state_name, self._buf_name, owner_pid, self._auto_unlink
        )

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return bool(int(self._state[_CLOSED]))

    @closed.setter
    def closed(self, val: bool) -> None:
        with self._state_lock:
            self._state[_CLOSED] = int(val)

    @property
    def nitems(self) -> int:
        with self._state_lock:
            return int(self._state[_NITEMS])

    @property
    def is_empty(self) -> bool:
        with self._state_lock:
            head = int(self._state[_HEAD])
            tail = int(self._state[_TAIL])
        return head == tail

    @property
    def bytes_available(self) -> int:
        with self._state_lock:
            head = int(self._state[_HEAD])
            tail = int(self._state[_TAIL])
        return (head - tail - 1) % self.cap

    def _wake_producers(self) -> None:
        """Wake every producer waiting for space. Caller holds _state_lock.

        Releasing inside the state lock keeps ``waiters + gate count == registered producers`` exact, which
        lets a producer that gives up waiting deregister without leaving a stray count (see DejaQueue.put).
        """
        w = int(self._state[_WAITERS])
        if w > 0:
            self._state[_WAITERS] = 0
            self._space_gate.release(w)

    def purge(self) -> None:
        """Clear all items from the queue."""
        with self._put_lock, self._get_lock:
            self._purge_locked()

    def _purge_locked(self) -> None:
        """Caller holds put_lock and get_lock."""
        with self._state_lock:
            self._state[_HEAD] = 0
            self._state[_TAIL] = 0
            self._state[_NITEMS] = 0
            while self._items.acquire(timeout=0):
                pass
            # a consumer that claimed an item before this purge sees NITEMS == 0 and waits again (see get)
            self._wake_producers()

    def _write_bytes(self, data, tail=None, write_tail=True) -> int:
        """Write data at current tail; return new tail (mod cap). Caller holds put_lock."""
        if not isinstance(data, memoryview):
            data = memoryview(data)
        n = len(data)
        cap = self.cap
        if tail is None:
            with self._state_lock:
                tail = int(self._state[_TAIL])
        end = tail + n
        if end <= cap:
            self.buf.buf[tail:end] = data
            new_tail = end % cap
        else:
            first = cap - tail
            self.buf.buf[tail:cap] = data[0:first]
            self.buf.buf[0 : n - first] = data[first:n]
            new_tail = n - first
        if write_tail:
            with self._state_lock:
                self._state[_TAIL] = new_tail
        return new_tail

    def _read_bytes(self, n: int, head: int | None = None) -> bytes:
        """Read n bytes from current head; advance head. Caller holds get_lock."""
        cap = self.cap
        if head is None:
            with self._state_lock:
                head = int(self._state[_HEAD])
        end = head + n
        if end <= cap:
            out = bytes(self.buf.buf[head:end])
            new_head = end % cap
        else:
            first = cap - head
            out = bytes(self.buf.buf[head:cap]) + bytes(self.buf.buf[0 : n - first])
            new_head = n - first
        with self._state_lock:
            self._state[_HEAD] = new_head
        return out

    def close(self) -> None:
        """Close access to shared memory and semaphores."""
        try:
            del self._state
            gc.collect(0)
        except Exception:
            pass
        for obj in [self._state_mem, self.buf, self._put_lock, self._get_lock, self._state_lock, self._items, self._space_gate]:
            try:
                obj.close()
            except Exception:
                pass

    def unlink(self) -> None:
        """Unlink shared memory and semaphores. Only call after all processes are done using the queue."""
        del self._state
        gc.collect(0)
        if self._owns:
            for obj in [self._state_mem, self.buf, self._put_lock, self._get_lock, self._state_lock, self._items, self._space_gate]:
                try:
                    obj.unlink()
                except Exception:
                    pass

    def __getstate__(self) -> dict:
        return {
            "base": self._base,
            "state_name": self._state_name,
            "buf_name": self._buf_name,
            "locks": (self._put_lock.__getstate__(), self._get_lock.__getstate__(), self._state_lock.__getstate__()),
            "items": self._items.__getstate__(),
            "space_gate": self._space_gate.__getstate__(),
            "auto_unlink": self._auto_unlink,
            "cap": self.cap,
        }

    def __setstate__(self, s: dict) -> None:
        self.__dict__.clear()
        self._owns = False
        self._auto_unlink = bool(s.get("auto_unlink", False))
        self._open(s["base"])  # all names derive deterministically from the base name
        self._register_finalizer()

    @staticmethod
    def _finalize(state_name: str, buf_name: str, owner_pid: int | None, auto_unlink: bool) -> None:
        gc.collect()
        # Only the creating process unlinks (not an attacher, nor a forked child that inherited the object).
        # Re-opening by name for anything else would be a no-op at best, and on POSIX < 3.13 would register
        # the segment with an attacher's own resource tracker, which then unlinks it when that process exits.
        if not (auto_unlink and owner_pid == os.getpid()):
            return
        for name in (state_name, buf_name):
            try:
                # Tracked open on purpose: unlink() then also drops the creator's registration.
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        try:
            if self._auto_unlink and self._owns and getattr(self, "_creator_pid", None) == os.getpid():
                self.unlink()
        except Exception:
            pass


class DejaQueue(NamedByteRing):
    """Pickleable queue for arbitrary Python objects. Uses a ring buffer of shared memory and named semaphores.

    Args:
        buffer_bytes (int): Size of the ring buffer in bytes. Default 10 MiB.
        name (str | None): Base name for shared memory and semaphores. If None (default), a random unique name is generated.
        create (bool): If True (default), create a new queue; raises FileExistsError if a queue with this name
            exists. If False, open an existing queue.
        auto_unlink (bool): If True (default), the creating process unlinks the shared memory and semaphores once
            its queue object is gone. Other processes must not attach after that.

    Example:
        >>> import pickle
        >>> q = DejaQueue(buffer_bytes=1e7)
        >>> pkl = pickle.dumps(q)
        >>> q2 = pickle.loads(pkl)
        >>> q.put({"a": 1, "b": [1,2,3]})
        >>> q2.get()
        {'a': 1, 'b': [1, 2, 3]}
    """

    def put(self, obj, timeout: float | None = None) -> bool:
        bufs = []
        p0 = _pickle_backend.dumps(obj, protocol=_PICKLE_PROTOCOL, buffer_callback=bufs.append)
        segs = [p0] + [b.raw() for b in bufs]
        lens = [len(s) for s in segs]
        hdr = _pack_header(lens)
        need = len(hdr) + sum(lens)

        if need >= self.cap:
            raise ValueError(
                f"Payload ({need} bytes) exceeds queue capacity ({self.cap} bytes). Increase buffer_bytes."
            )

        deadline = None if timeout is None else (time.monotonic() + float(timeout))
        while True:
            with self._put_lock:
                st = self._state  # hot path: literal slot indices (see _HEAD, _TAIL, ... above)
                with self._state_lock:
                    head = st[0]  # _HEAD
                    tail = st[1]  # _TAIL
                    fits = (head - tail - 1) % self.cap >= need
                    if not fits:
                        # Register as waiting while still holding the state lock: any consumer that frees space
                        # after the check above sees the registration and wakes us (no lost wake-up).
                        st[5] += 1  # _WAITERS
                if fits:
                    new_tail = self._write_bytes(hdr, tail=tail, write_tail=False)
                    for s in segs:
                        new_tail = self._write_bytes(s, tail=new_tail, write_tail=False)
                    with self._state_lock:
                        st[1] = new_tail  # _TAIL
                        st[2] += 1  # _NITEMS
                    self._items.release(1)
                    return True

            try:
                if deadline is None:
                    woken = self._space_gate.acquire()
                else:
                    rem = deadline - time.monotonic()
                    woken = rem > 0 and self._space_gate.acquire(timeout=rem)
            except BaseException:  # e.g. KeyboardInterrupt while waiting
                self._cancel_space_wait()
                raise
            if not woken:
                self._cancel_space_wait()
                raise TimeoutError("Timeout waiting for space in queue.")

    def _cancel_space_wait(self) -> None:
        """Deregister a producer that stops waiting for space without having been woken."""
        with self._state_lock:
            if int(self._state[_WAITERS]) > 0:
                self._state[_WAITERS] = int(self._state[_WAITERS]) - 1
            else:
                # A consumer already turned all registrations into gate counts (inside the state lock), so
                # one of them is ours: take it back. It cannot be missing, as waiters + gate count always
                # equals the number of registered producers, and we are still one of them.
                self._space_gate.acquire(timeout=0)

    def get(self, timeout=None, peek_only=False, callback=None):
        """Gets an item from the queue.

        Args:
            timeout (float | None): Maximum time to wait for an item. Default None (wait indefinitely).
            peek_only (bool): If True, read the item without removing it from the queue. Default False.
            callback (Callable, optional): If provided, a function to be called with the unpickled object
                (pre-copy, potentially unsafe!). If None (default), a deepcopy of the object is returned.
                Make sure the callback is fast, since it runs while holding the get_lock, and that it does not keep references
                to the object (or its buffers), since the underlying memory may be overwritten.
        """
        deadline = None if timeout is None else (time.monotonic() + max(0.0, float(timeout)))
        while True:
            # Claim an item first, then take the get lock (as queue.Queue does). Nobody waits for an item while
            # holding the lock, so a get with a timeout is never stuck behind another consumer's wait; the lock
            # itself is only held while an item is read.
            rem = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not self._items.acquire(timeout=rem):
                raise TimeoutError("Timeout waiting for item.")
            try:
                self._get_lock.acquire()
            except BaseException:
                self._items.release(1)  # give the claim back
                raise
            try:
                st = self._state  # hot path: literal slot indices
                with self._state_lock:
                    nitems = st[2]  # _NITEMS
                    head0 = st[0]  # _HEAD
                if nitems > 0:
                    return self._get_locked(head0, peek_only, callback)
                # Our claim predates a purge() that removed the item: wait for a new one.
            finally:
                self._get_lock.release()

    def _get_locked(self, head0: int, peek_only: bool, callback):
        """Read the item at head0. Caller holds get_lock and one count of _items."""
        cap = self.cap
        buf = self.buf.buf

        def _copy_span(start, n):
            start %= cap
            end = start + n
            if end <= cap:
                return bytes(buf[start:end])
            first = cap - start
            out = b"".join((buf[start:cap], buf[0 : n - first]))
            return out

        # header
        K = struct.unpack("<Q", _copy_span(head0, 8))[0]
        lens = struct.unpack(f"<{K}Q", _copy_span(head0 + 8, 8 * K))
        total = 8 + 8 * K + sum(lens)
        segs, cursor = [], head0 + 8 + 8 * K

        for n in lens:
            s = cursor % cap
            e = s + n
            if e <= cap:
                segs.append(buf[s:e])
            else:
                segs.append(_copy_span(cursor, n))  # wrapped -> copy
            cursor += n

        try:
            obj = _pickle_backend.loads(segs[0], buffers=[m for m in segs[1:]])
            out = deepcopy(obj) if callback is None else callback(obj)
        except BaseException as e:
            # Never leave a failing item at the head with its count consumed: every later get()
            # would fail on it again. A get that fails while unpickling/copying/in the callback
            # drops the item (as a successful get would); a failed peek, or an interrupt
            # (KeyboardInterrupt, SystemExit), leaves it in the queue.
            segs = obj = None  # release views of the ring before its space is reused
            self._finish_get(head0, total, consume=isinstance(e, Exception) and not peek_only)
            raise
        self._finish_get(head0, total, consume=not peek_only)  # advance only after loads()
        return out

    def _finish_get(self, head0: int, total: int, consume: bool) -> None:
        """Caller holds get_lock and one count of _items. Drop the item at head0, or put the count back."""
        if consume:
            st = self._state  # hot path: literal slot indices
            with self._state_lock:
                st[0] = (head0 + total) % self.cap  # _HEAD
                st[2] -= 1  # _NITEMS
                if st[5]:  # _WAITERS
                    self._wake_producers()
        else:
            self._items.release(1)

    def __iter__(self):
        while True:
            x = self.get()
            if x is Ellipsis:
                self.put(Ellipsis)
                return
            yield x

    def _signal_stop(self, n=1):
        """Puts n stop signals into the queue."""
        for _ in range(n):
            self.put(Ellipsis)


### Legacy DejaQueue: ###

class ByteFIFO:
    """A FIFO buffer (queue) for bytes. The queue is implemented as a ring buffer in shared memory."""

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
            # array_bytes = np.frombuffer(array_bytes, dtype='byte')
            array_bytes = array_bytes
        elif type(array_bytes) == np.ndarray:
            array_bytes = array_bytes.ravel().view("B")
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
        """Write a byte array into the queue. Warning: this function should be called after acquiring the put_lock."""
        old_tail = old_tail or self.tail.value
        nbytes = len(array_bytes)
        if old_tail + nbytes <= self.buffer_bytes:
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
        """Gets a byte array from the queue.

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
        """Calculates the available space in the buffer.

        Returns:
            int: The available space in bytes.
        """
        return (self.head.value - self.tail.value - 1) % self.buffer_bytes

    @property
    def view(self):
        """numpy.ndarray: A view of the shared memory array as a numpy array. Lazy initialization to avoid pickling issues."""
        if self._view is None:
            self._view = np.frombuffer(self.buffer, "B")
        return self._view

    def __del__(self):
        self._view = None

    def empty(self):
        """Checks if the queue is empty.

        Returns:
            bool: True if the queue is empty, False otherwise.
        """
        return self.queue.empty()

    def close(self):
        """Closes the queue."""
        self.closed.value = True

    def join(self):
        """Joins the queue"""
        self.queue.join()

    def task_done(self):
        """Marks a task as done."""
        self.queue.task_done()

    @property
    def done(self):
        """Returns True if the queue is empty and closed."""
        return self.queue.empty() and self.closed.value

    def __iter__(self):
        while True:
            x = self.get()
            if x is Ellipsis:
                self.queue.put(Ellipsis)
                return
            yield x

    def _signal_stop(self, n=1):
        """Puts n stop signals into the queue."""
        for _ in range(n):
            self.queue.put(Ellipsis)

    def __getstate__(self):
        state = {k: v for k, v in self.__dict__.items() if k != "_view"}
        state["_view"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class LegacyDejaQueue(ByteFIFO):
    """A fast queue for arbitrary (picklable) Python objects.

    Args:
        buffer_bytes (int): The size of the buffer in bytes. Defaults to 10 MiB.
    """

    def __init__(self, buffer_bytes=10e6):
        super().__init__(buffer_bytes=buffer_bytes)

    def put(self, obj, timeout=None):
        """Puts a Python object into the queue.

        Args:
            obj (Any): The byte array to be put into the queue.
            timeout (float, optional): The maximum time to wait for available space in the queue.
        """
        buffers = []
        pkl = _pickle_backend.dumps(obj, buffer_callback=buffers.append, protocol=_PICKLE_PROTOCOL)
        buffer_lengths = [len(pkl)] + [len(it.raw()) for it in buffers]
        nbytes_total = sum(buffer_lengths)

        assert nbytes_total < self.buffer_bytes, "Array size exceeds buffer size."

        with self.put_lock:
            while self._available_space() < nbytes_total:
                with self.head_changed:
                    if not self.head_changed.wait(timeout=timeout):
                        raise TimeoutError("Timeout waiting for available space.")

            head = self.tail.value
            self._write_buffer(np.frombuffer(pkl, "byte"))
            for buf in buffers:
                self._write_buffer(buf.raw())

            frame_info = FrameInfo(nbytes=nbytes_total, head=head, tail=self.tail.value, meta=buffer_lengths)
            self.queue.put(frame_info)

    def get(self, **kwargs):
        """Gets an item from the queue.

        Args:
            **kwargs: Additional keyword arguments to be passed to the underlying queue's get method (e.g. timeout).

        Returns:
            obj: The object that was put into the queue.
        """

        def callback(array_bytes, buffer_lengths):
            buffers = []
            offset = 0
            for length in buffer_lengths:
                buffers.append(_pickle_backend.PickleBuffer(array_bytes[offset : offset + length]))
                offset += length
            obj = _pickle_backend.loads(buffers[0], buffers=buffers[1:])
            return obj

        obj = super().get(copy=False, callback=callback, **kwargs)
        return obj


@dataclasses.dataclass
class FrameInfo:
    """A class to store metadata about a data frame in a ring buffer."""

    nbytes: int
    head: int
    tail: int
    meta: Any  # any picklable object
