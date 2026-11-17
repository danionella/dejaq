from __future__ import annotations
import logging
import atexit, collections, contextlib, errno, gc, numbers, os, sys, time, warnings, weakref, struct, threading
import queue as _stdqueue
import multiprocessing as mp
from multiprocessing import shared_memory
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
    _CreateMutexW = _kernel32.CreateMutexW
    _CreateMutexW.argtypes = [ctypes.c_void_p, _W.BOOL, _W.LPCWSTR]
    _CreateMutexW.restype = _W.HANDLE
    _OpenMutexW = _kernel32.OpenMutexW
    _OpenMutexW.argtypes = [_W.DWORD, _W.BOOL, _W.LPCWSTR]
    _OpenMutexW.restype = _W.HANDLE
    _ReleaseMutex = _kernel32.ReleaseMutex
    _ReleaseMutex.argtypes = [_W.HANDLE]
    _ReleaseMutex.restype = _W.BOOL
    _WaitForSingleObject = _winapi.WaitForSingleObject  # raises OSError on WAIT_FAILED
    _CloseHandle = _winapi.CloseHandle

    _INFINITE = _winapi.INFINITE
    _WAIT_OBJECT_0 = _winapi.WAIT_OBJECT_0
    _WAIT_TIMEOUT = _winapi.WAIT_TIMEOUT
    _SEMAPHORE_MODIFY_STATE = 0x0002
    _MUTEX_MODIFY_STATE = 0x0001
    _WAIT_ABANDONED = 0x80  # mutex acquired, but its previous owner exited without releasing it
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
                raise ValueError("operation on closed lock or semaphore")
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


# The resource tracker must not be called by a finalizer that the garbage collector runs while the tracker
# itself is busy in this thread: Python >= 3.11.7 / 3.12.1 then skips the call with a warning, earlier
# versions deadlock. So finalizers run by the garbage collector only queue their tracker calls, which are
# sent with the next call from regular code (or at exit); and dejaq's own tracker calls pause the garbage
# collector, so that it cannot run finalizers (of dejaq or anything else) inside them.
#
# All calls go through one queue, sent in order by one thread at a time (a later "unregister" must never
# overtake its "register"). Calls queued while this thread runs the garbage collector, or while it is
# sending the queue itself (e.g. from a finalizer run by refcounting), are sent by the sending loop.
_tracker_backlog: "collections.deque" = collections.deque()
_tracker_send_lock = threading.Lock()
_tracker_local = threading.local()  # .in_gc: this thread runs the garbage collector; .sending: see above


def _note_gc(phase: str, info: dict) -> None:
    _tracker_local.in_gc = phase == "start"  # gc callbacks run in the collecting thread


gc.callbacks.append(_note_gc)

# Pausing the garbage collector is process-wide; nested and concurrent pauses (several threads creating
# queues) are counted, and only the end of the last one restores the previous state.
_gc_pause_lock = threading.Lock()
_gc_pause_depth = 0
_gc_pause_restore = False


@contextlib.contextmanager
def _gc_paused():
    global _gc_pause_depth, _gc_pause_restore
    with _gc_pause_lock:
        if _gc_pause_depth == 0:
            _gc_pause_restore = gc.isenabled()
            gc.disable()
        _gc_pause_depth += 1
    try:
        yield
    finally:
        with _gc_pause_lock:
            _gc_pause_depth -= 1
            if _gc_pause_depth == 0 and _gc_pause_restore:
                gc.enable()


def _tracker(action: str, name: str, rtype: str) -> None:
    """Register/unregister a POSIX resource with this process's multiprocessing resource tracker, which
    removes still-registered resources when the process tree's main process ends (also after a crash)."""
    _tracker_backlog.append((action, name, rtype))
    if not getattr(_tracker_local, "in_gc", False) and not getattr(_tracker_local, "sending", False):
        _flush_tracker_backlog()


def _flush_tracker_backlog() -> None:
    if not _tracker_backlog or getattr(_tracker_local, "sending", False):  # (the sending loop sends it)
        return
    from multiprocessing import resource_tracker

    with _tracker_send_lock, _gc_paused():
        _tracker_local.sending = True
        try:
            while True:
                try:
                    action, name, rtype = _tracker_backlog.popleft()
                except IndexError:
                    break
                try:
                    getattr(resource_tracker, action)(name, rtype)
                except Exception:  # crash cleanup is best effort
                    pass
        finally:
            _tracker_local.sending = False


atexit.register(_flush_tracker_backlog)  # calls queued by the last garbage collections
if not _IS_WIN:
    # multiprocessing workers end with os._exit, which skips atexit; multiprocessing's own exit handlers run
    # its "finalizers" instead. A worker clears the finalizers it inherited or registered on import before
    # it runs, and then calls the after-fork hooks: register (again) there.
    from multiprocessing import util as _mp_util

    class _Anchor:  # register_after_fork needs an object to refer to
        pass

    _mp_anchor = _Anchor()

    def _flush_at_worker_exit(_anchor=None) -> None:
        _mp_util.Finalize(None, _flush_tracker_backlog, exitpriority=0)

    _flush_at_worker_exit()
    _mp_util.register_after_fork(_mp_anchor, _flush_at_worker_exit)


def _reset_tracker_state_in_child() -> None:
    """After fork: the parent sends its own queued calls, and locks held by its other threads stay held."""
    global _tracker_send_lock, _gc_pause_lock, _gc_pause_depth
    _tracker_backlog.clear()
    _tracker_send_lock = threading.Lock()
    _gc_pause_lock = threading.Lock()
    if _gc_pause_depth:  # another thread of the parent was inside a pause: it does not exist here
        _gc_pause_depth = 0
        if _gc_pause_restore:
            gc.enable()


# POSIX lock files live in a fixed directory (not $TMPDIR, which may differ between processes): every
# process must find the same file. On Linux /dev/shm is a tmpfs, like POSIX shared memory itself.
_LOCK_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
# glibc's shm_unlink(name) removes /dev/shm/<name>, so on Linux the resource tracker can clean up lock
# files registered as shared memory after a crash. Elsewhere, files of crashed queues remain (empty).
_LOCK_FILES_TRACKABLE = sys.platform.startswith("linux") and _LOCK_DIR == "/dev/shm"


_FD_NOT_REOPENED = -2  # a lock whose file could not be reopened in a forked child


class _RobustLock:
    """Cross-process mutex that the operating system releases when its holder dies, however it dies.

    POSIX: flock() on a lock file. flock locks belong to an open file description, so each object opens its
    own (and reopens it in a forked child, which would otherwise share the parent's), and a threading.Lock
    serializes threads sharing one object. Windows: a named mutex (released on exit as "abandoned").

    The lock itself does not report whether a previous holder died; callers that need to know keep a flag
    in shared memory (see DejaQueue.get).
    """

    _posix_instances: "weakref.WeakSet[_RobustLock]" = weakref.WeakSet()

    def __init__(self, name: str, create: bool = True, *, auto_unlink: bool = True) -> None:
        self.name = name
        self._owner_pid = os.getpid() if create else None
        self._tracked = [False]  # shared with the finalizer: registered with the resource tracker (POSIX)
        if _IS_WIN:
            nm = _win_name(name)
            ctypes.set_last_error(0)
            if create:
                h = _CreateMutexW(None, False, nm)
                if h and ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
                    _CloseHandle(h)
                    raise FileExistsError(errno.EEXIST, "lock already exists", nm)
            else:
                h = _OpenMutexW(_SYNCHRONIZE | _MUTEX_MODIFY_STATE, False, nm)
            if not h:
                raise ctypes.WinError(ctypes.get_last_error())
            self._h = _WinHandle(h)
            handle = self._h
        else:
            self.path = _RobustLock._path_for(name)
            flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
            self._fd = [os.open(self.path, flags, 0o600)]  # a box: replaced after fork
            self._tlock = threading.Lock()
            if create and auto_unlink and _LOCK_FILES_TRACKABLE:
                _tracker("register", "/" + os.path.basename(self.path), "shared_memory")
                self._tracked[0] = True
            _RobustLock._posix_instances.add(self)
            handle = self._fd
        self._finalizer = weakref.finalize(
            self, _RobustLock._finalize, handle, getattr(self, "path", None), self._owner_pid, auto_unlink, self._tracked
        )

    @staticmethod
    def _path_for(name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.lstrip("/"))
        return os.path.join(_LOCK_DIR, "dejaq." + safe + ".lock")

    def acquire(self) -> None:
        if _IS_WIN:
            rc = _WaitForSingleObject(self._h.get(), _INFINITE)
            if rc != _WAIT_OBJECT_0 and rc != _WAIT_ABANDONED:
                raise RuntimeError(f"WaitForSingleObject rc={rc}")
            return
        self._tlock.acquire()
        try:
            _flock(self._fd[0], _LOCK_EX)
        except BaseException:
            self._tlock.release()
            if self._fd[0] == _FD_NOT_REOPENED:
                raise RuntimeError(
                    "This queue was unlinked before this process was forked, and its lock cannot be reopened "
                    "here; create or attach queues after forking, or unlink them later."
                ) from None
            raise

    def release(self) -> None:
        if _IS_WIN:
            if not _ReleaseMutex(self._h.get()):
                raise RuntimeError("release of a lock that is not held")
            return
        _flock(self._fd[0], _LOCK_UN)
        self._tlock.release()  # raises RuntimeError if the lock was not held

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()

    def close(self) -> None:
        if _IS_WIN:
            self._h.close()
        else:
            fd, self._fd[0] = self._fd[0], -1
            if fd >= 0:
                os.close(fd)

    def unlink(self) -> None:
        """Remove the lock's name (POSIX: its file). Holders of the lock are not affected."""
        _RobustLock._unlink(getattr(self, "path", None), self._owner_pid, self._tracked)

    @staticmethod
    def _unlink(path, owner_pid, tracked) -> None:
        if path is None:  # Windows: the mutex disappears with its last handle
            return
        try:
            os.unlink(path)
        except OSError:
            pass
        if tracked[0] and owner_pid == os.getpid():
            tracked[0] = False
            _tracker("unregister", "/" + os.path.basename(path), "shared_memory")

    @staticmethod
    def _finalize(handle, path, owner_pid, auto_unlink, tracked) -> None:
        try:
            if _IS_WIN:
                handle.close()
            else:
                fd, handle[0] = handle[0], -1
                if fd >= 0:
                    os.close(fd)
            if auto_unlink and owner_pid == os.getpid():
                _RobustLock._unlink(path, owner_pid, tracked)
        except Exception:
            pass

    @classmethod
    def _after_fork_in_child(cls) -> None:
        # The child got copies of the parent's file descriptors, which share the parent's open file
        # descriptions and hence its flock locks. Give every object its own again.
        for lk in list(cls._posix_instances):
            old = lk._fd[0]
            if old >= 0:
                try:
                    new_fd = os.open(lk.path, os.O_RDWR)
                except OSError:  # the lock file was deleted (unlink()): reopen the open file itself
                    try:
                        new_fd = os.open(f"/proc/self/fd/{old}", os.O_RDWR)  # Linux
                    except OSError:
                        new_fd = _FD_NOT_REOPENED  # never share the parent's lock: acquire() raises
                lk._fd[0] = new_fd
                os.close(old)
            lk._tlock = threading.Lock()  # may have been held by another thread of the parent


if not _IS_WIN:
    import fcntl

    _flock, _LOCK_EX, _LOCK_UN = fcntl.flock, fcntl.LOCK_EX, fcntl.LOCK_UN
    os.register_at_fork(after_in_child=_RobustLock._after_fork_in_child)
    os.register_at_fork(after_in_child=_reset_tracker_state_in_child)


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
        self._owner_pid = owner_pid = os.getpid() if create else None
        # POSIX semaphores outlive crashed processes: let the resource tracker remove them in that case
        self._tracked = [bool(create and self._auto_unlink and not _IS_WIN)]
        if self._tracked[0]:
            _tracker("register", self.name, "semaphore")
        self._finalizer = weakref.finalize(
            self, NamedSemaphore._finalize, self.backend, self.name, self._auto_unlink, owner_pid, handle,
            self._tracked,
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
            if timeout is None:
                self._sem.acquire()
                return True
            import posix_ipc as P

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
            sem = self._sem
            if n == 1:
                sem.release()
                return
            for _ in range(int(n)):
                sem.release()

    def close(self) -> None:
        self._finalizer()

    def unlink(self) -> None:
        if not _IS_WIN:
            NamedSemaphore._unlink(self.name, self._owner_pid, self._tracked)

    @staticmethod
    def _unlink(name: str, owner_pid: int | None, tracked: list) -> None:
        import posix_ipc as P

        try:
            P.unlink_semaphore(name)
        except Exception:
            pass
        if tracked[0] and owner_pid == os.getpid():
            tracked[0] = False
            _tracker("unregister", name, "semaphore")

    def __getstate__(self) -> dict:
        return {"name": self.name, "backend": self.backend, "auto_unlink": self._auto_unlink}

    def __setstate__(self, s: dict) -> None:
        self.__dict__.clear()
        self.backend = s["backend"]
        auto_unlink = bool(s.get("auto_unlink", False))
        self.__init__(s["name"], create=False, auto_unlink=auto_unlink)

    @staticmethod
    def _finalize(
        backend: str, name: str, auto_unlink: bool, owner_pid: int | None, handle: Any, tracked: list
    ) -> None:
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
                    NamedSemaphore._unlink(name, owner_pid, tracked)
                    logger.debug(f"finalizer unlinking {name}")
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


class _StateLock(NamedSemaphore):
    """Internal mutex for short critical sections: like NamedLock, but releasing is a single operation
    (no over-release probe on POSIX; Windows still enforces maxcount=1). Only for strictly paired use."""

    def __init__(self, name: str | None = None, create: bool = True, *, auto_unlink: bool = True) -> None:
        super().__init__(name=name, create=create, initial=1, maxcount=1, auto_unlink=auto_unlink)

    if not _IS_WIN:

        def release(self) -> None:
            self._sem.release()

        def __enter__(self):
            self._sem.acquire()
            return self

        def __exit__(self, *_):
            self._sem.release()


class _SharedMemory(shared_memory.SharedMemory):
    """SharedMemory that owns the views derived from it and releases them in close() (and tolerates views that
    others still hold, see close()).

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
        try:
            super().close()
        except BufferError:
            # Arrays handed out without a copy (get(callback=...), dejaq.parallel's zero-copy iteration) still
            # view the memory: leave the mapping to them (it is unmapped when the last of them is gone).
            self._buf = self._mmap = None
            fd = getattr(self, "_fd", -1)
            if fd >= 0:
                os.close(fd)
                self._fd = -1


# Shared state block: int64 slots, read and written only under _state_lock.
#   _WAITERS counts producers registered as waiting for space (see DejaQueue.put).
#   _MAGIC identifies the layout of this block and of the items in the ring: processes sharing a queue
#   must run compatible dejaq versions, and a mismatch is reported instead of corrupting data.
#   _GET_CLAIM is set while the get lock's holder holds a claimed item (see DejaQueue.get).
_HEAD, _TAIL, _NITEMS, _CLOSED, _CAP, _WAITERS, _MAGIC, _GET_CLAIM = range(8)
_STATE_SLOTS = 8
_STATE_MAGIC = 0x64656A61710003  # "dejaq" + format version 3 (1: dejaq <= 0.8; 2: without _GET_CLAIM)


def _untrack_shm(shm) -> None:
    """Keep a segment beyond this process: the resource tracker would otherwise remove it at exit."""
    if not _IS_WIN:
        _tracker("unregister", shm._name, "shared_memory")


def _unlink_shm(shm, tracked: bool) -> None:
    if _IS_WIN:  # the segment disappears with its last handle
        return
    import _posixshmem

    try:
        _posixshmem.shm_unlink(shm._name)  # (not shm.unlink(): it calls the resource tracker directly)
    except FileNotFoundError:
        pass
    if tracked:
        _tracker("unregister", shm._name, "shared_memory")


class _RingResources:
    """Everything one ring object holds open. close() and unlink() are idempotent; finalize() runs once the
    ring object is gone (or the queue is used as a context manager): it closes everything, and in the
    creating process of a queue with auto_unlink it also removes the names."""

    def __init__(self, shms, others, owner_pid, auto_unlink: bool, shm_tracked: bool) -> None:
        self.shms, self.others = list(shms), list(others)
        self.owner_pid, self.auto_unlink, self.shm_tracked = owner_pid, auto_unlink, shm_tracked
        self.unlinked = False

    def close(self) -> None:
        for obj in (*self.shms, *self.others):
            try:
                obj.close()  # _SharedMemory.close also releases the views derived from it
            except Exception:
                pass

    def unlink(self) -> None:
        if self.unlinked:
            return
        self.unlinked = True
        first_error = None
        for obj in self.shms + self.others:  # remove everything that can be removed
            try:
                if obj in self.shms:
                    _unlink_shm(obj, self.shm_tracked)
                else:
                    obj.unlink()
            except Exception as e:
                first_error = first_error or e
        if first_error is not None:
            raise first_error

    def finalize(self) -> None:
        self.close()
        if self.auto_unlink and self.owner_pid == os.getpid():  # not in a forked child, nor an attacher
            try:
                self.unlink()
            except Exception as e:  # no caller to report to
                logger.warning(f"could not remove all of a queue's shared memory and semaphores: {e!r}")


class Empty(_stdqueue.Empty, TimeoutError):
    """Raised by DejaQueue.get() when no item is available in time (also a queue.Empty and a TimeoutError)."""


class Full(_stdqueue.Full, TimeoutError):
    """Raised by DejaQueue.put() when there is no space in time (also a queue.Full and a TimeoutError)."""


class ShutDown(getattr(_stdqueue, "ShutDown", Exception)):
    """Raised by put() on a queue that was shut down, and by get() once a shut-down queue is empty
    (a queue.ShutDown on Python >= 3.13)."""


def _is_number(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _wait_time(block, timeout, method: str):
    """Timeout of get()/put() from queue.Queue-style (block, timeout) arguments (None: wait indefinitely).

    dejaq <= 0.9 took the timeout as the first argument; a number there is still taken as a timeout
    (deprecated). get(None) waited indefinitely then and still does.
    """
    if block is None:
        return timeout
    if _is_number(block):
        if timeout is not None:
            raise TypeError(f"DejaQueue.{method}(): block must be True or False, got {block!r}")
        warnings.warn(
            f"DejaQueue.{method}(): a timeout as first argument is deprecated; pass timeout=... as a keyword "
            "(positional arguments now follow queue.Queue: block, timeout)",
            DeprecationWarning,
            stacklevel=3,
        )
        return block
    return timeout if block else 0.0


def _pack_header(lens) -> bytes:
    """Item header: number of segments K, then K segment lengths (all uint64)."""
    return struct.pack(f"<Q{len(lens)}Q", len(lens), *lens)


_MIN_ITEM_BYTES = 16 + len(pickle.dumps(None, protocol=5))  # header with one segment + the smallest pickle


class NamedByteRing:
    """Manager/Condition-free ring buffer queue (bytes) in shared memory.
       Pickleable across independently started processes.

    A queue keeps working when a process using it dies, even while holding the put or get lock (the
    operating system releases those locks). An item whose consumer died while reading it is delivered again.

    Args:
        buffer_bytes (int): Size of the ring buffer in bytes.
        name (str | None): Base name for shared memory and semaphores. If None (default), a random unique name is generated.
        create (bool): If True (default), create new shared memory and semaphores; raises FileExistsError if a
            queue with this name exists. If False, open an existing queue.
        auto_unlink (bool): If True (default), the queue is removed once the creating process's queue object is
            gone (or that process ends, also by crashing): processes already using it can continue, but no new
            process can attach. If False, the queue persists until ``unlink()`` is called (POSIX; on Windows a
            queue always exists exactly as long as some process has it open).

    Can be used as a context manager: ``with DejaQueue() as q: ...`` closes it at the end (and removes it, if
    this object created it with auto_unlink).
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
        self._finalizer = weakref.finalize(self, self._res.finalize)

    @staticmethod
    def _names(base: str):
        """Names of (state shm, buffer shm, [put lock, get lock], [state lock, items, space gate]) for a base name."""
        if _IS_WIN:  # mutexes and semaphores share one namespace: distinct prefixes
            return ("NS_" + base, "NB_" + base, ["NXp_" + base, "NXg_" + base],
                    [p + base for p in ("NLs_", "NI_", "NG_")])
        return (_shm_name(base + "_S"), _shm_name(base + "_B"), [base + "_Lp", base + "_Lg"],
                [base + s for s in ("_Ls", "_I", "_G")])

    def _create_named(self, cls, name: str):
        try:
            return cls(name, create=True, auto_unlink=self._auto_unlink)
        except FileExistsError:
            if _IS_WIN:
                raise
        # POSIX semaphores and lock files outlive the processes that created them, e.g. after a crash. The
        # caller has just created this queue's shared memory exclusively, so a semaphore or lock file under
        # its name can only belong to a dead queue: replace it. (Processes still attached to the dead queue
        # keep their own.)
        if cls is _RobustLock:
            try:
                os.unlink(_RobustLock._path_for(name))
            except OSError:
                pass
        else:
            import posix_ipc as P

            try:
                P.unlink_semaphore(_posix_name(name))
            except P.ExistentialError:
                pass
        return cls(name, create=True, auto_unlink=self._auto_unlink)

    def _create(self, base: str, buffer_bytes: int) -> None:
        st_name, buf_name, lock_names, sem_names = self._names(base)
        shms, others = [], []  # removed again if any step fails
        shm_tracked = True  # SharedMemory(create=True) registers with the resource tracker
        try:
            with _gc_paused():
                # SharedMemory registers directly: send queued calls first, e.g. an "unregister" of the
                # same name by a finalized queue, which must not follow (and cancel) this registration
                _flush_tracker_backlog()
                state_mem = _SharedMemory(create=True, name=st_name, size=8 * _STATE_SLOTS)
                shms.append(state_mem)
                buf = _SharedMemory(name=buf_name, create=True, size=buffer_bytes)
                shms.append(buf)
            if not self._auto_unlink:
                for shm in shms:
                    _untrack_shm(shm)
                shm_tracked = False
            _view = np.frombuffer(buf.buf, dtype="B", count=buffer_bytes)
            _view[:] = 0
            del _view
            for nm in lock_names:
                others.append(self._create_named(_RobustLock, nm))
            for cls, nm in zip((_StateLock, NamedSemaphore, NamedSemaphore), sem_names):
                others.append(self._create_named(cls, nm))
            state = state_mem.cast("q")
            state[0:_STATE_SLOTS] = array.array("q", [0, 0, 0, 0, buffer_bytes, 0, _STATE_MAGIC, 0])
        except BaseException:
            res = _RingResources(shms, others, os.getpid(), True, shm_tracked)
            res.close()
            try:
                res.unlink()
            except Exception:
                pass  # report the original error
            raise
        self._base = base
        self._state_mem, self._state, self.buf, self.cap = state_mem, state, buf, buffer_bytes
        self._state_name, self._buf_name = st_name, buf_name
        self._put_lock, self._get_lock, self._state_lock, self._items, self._space_gate = others
        self._owns = True
        self._res = _RingResources(shms, others, os.getpid(), self._auto_unlink, self._auto_unlink)

    def _open(self, base: str) -> None:
        st_name, buf_name, lock_names, sem_names = self._names(base)
        self._base = base
        shms, others = [], []
        try:
            try:
                shms.append(_attach_shm(st_name, cls=_SharedMemory))
            except FileNotFoundError:
                hint = (
                    "on Windows, a queue exists only while some process has it open"
                    if _IS_WIN
                    else "with auto_unlink=True (the default), a queue is removed when its creating object is "
                    "garbage-collected"
                )
                raise FileNotFoundError(
                    errno.ENOENT, f"DejaQueue {base!r} does not exist (any more): {hint}. Keep it alive while "
                    "processes attach."
                ) from None
            state = shms[0].cast("q")
            if len(state) < _STATE_SLOTS or state[_MAGIC] != _STATE_MAGIC:
                raise RuntimeError(
                    f"Queue {base!r} was created by an incompatible dejaq version; "
                    "all processes sharing a queue must use the same dejaq version."
                )
            shms.append(_attach_shm(buf_name, cls=_SharedMemory))
            for nm in lock_names:
                others.append(_RobustLock(nm, create=False))
            others.append(_StateLock(sem_names[0], create=False))
            for nm in sem_names[1:]:
                others.append(NamedSemaphore(nm, create=False))
        except BaseException:
            _RingResources(shms, others, None, False, False).close()
            raise
        self._state_mem, self._state, self.buf = shms[0], state, shms[1]
        self.cap = int(state[_CAP])  # read cap stored by the creator
        self._state_name, self._buf_name = st_name, buf_name
        self._put_lock, self._get_lock, self._state_lock, self._items, self._space_gate = others
        self._res = _RingResources(shms, others, None, False, False)

    @property
    def is_shutdown(self) -> bool:
        """True once shutdown() was called (by any process)."""
        return bool(self._state[_CLOSED])

    @property
    def closed(self) -> bool:
        """Alias of is_shutdown (setting it to True shuts the queue down)."""
        return self.is_shutdown

    @closed.setter
    def closed(self, val: bool) -> None:
        if val:
            self.shutdown()
        else:  # reopen
            with self._state_lock:
                self._state[_CLOSED] = 0

    def shutdown(self, immediate: bool = False) -> None:
        """Shut the queue down, for all processes using it (like queue.Queue.shutdown in Python 3.13).

        Afterwards put() raises ShutDown, and get() returns the remaining items and then raises ShutDown.
        Producers and consumers waiting at that moment are woken and raise ShutDown (consumers only once
        the queue is empty). With immediate=True, the remaining items are dropped as well.
        """
        if immediate:
            with self._put_lock, self._get_lock:
                with self._state_lock:
                    self._state[_CLOSED] = 1
                self._purge_locked()  # also wakes waiting producers and consumers
        else:
            with self._state_lock:
                self._state[_CLOSED] = 1
                self._wake_producers()
            # Wake waiting consumers: each one that finds the queue shut down and empty passes this count on.
            self._items.release(1)

    @property
    def nitems(self) -> int:
        """Number of items in the queue."""
        with self._state_lock:
            return int(self._state[_NITEMS])

    @property
    def is_empty(self) -> bool:
        with self._state_lock:
            head = int(self._state[_HEAD])
            tail = int(self._state[_TAIL])
        return head == tail

    @property
    def bytes_free(self) -> int:
        """Free space in the ring buffer, in bytes (an item needs its pickled size plus a small header)."""
        with self._state_lock:
            head = int(self._state[_HEAD])
            tail = int(self._state[_TAIL])
        return (head - tail - 1) % self.cap

    bytes_available = bytes_free  # former name

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
        self._state[_GET_CLAIM] = 0  # all claims are void (and an abandoned one needs no repair)
        with self._state_lock:
            self._state[_HEAD] = 0
            self._state[_TAIL] = 0
            self._state[_NITEMS] = 0
            while self._items.acquire(timeout=0):
                pass
            # counts handed back later by waiting consumers are spare: they are dropped (see get)
            self._wake_producers()
            if self._state[_CLOSED]:  # keep shutdown()'s wake-up for consumers waiting on the empty queue
                self._items.release(1)

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
        """Close this object's access to the queue (other objects and processes are not affected)."""
        self._res.close()

    def unlink(self) -> None:
        """Remove the queue's names, so that no further process can attach. Processes that already use the
        queue can continue. Any process may call this, e.g. to remove a queue created with auto_unlink=False."""
        self._res.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self._finalizer()  # close, and remove the queue if this object created it with auto_unlink

    def __getstate__(self) -> dict:
        return {"base": self._base, "auto_unlink": self._auto_unlink, "cap": self.cap}

    def __setstate__(self, s: dict) -> None:
        self.__dict__.clear()
        self._owns = False
        self._auto_unlink = bool(s.get("auto_unlink", False))
        self._open(s["base"])  # all names derive deterministically from the base name
        self._finalizer = weakref.finalize(self, self._res.finalize)


class DejaQueue(NamedByteRing):
    """Pickleable queue for arbitrary Python objects. Uses a ring buffer of shared memory and named semaphores.

    Args:
        buffer_bytes (int): Size of the ring buffer in bytes. Default 10 MiB.
        name (str | None): Base name for shared memory and semaphores. If None (default), a random unique name is generated.
        create (bool): If True (default), create a new queue; raises FileExistsError if a queue with this name
            exists. If False, open an existing queue.
        auto_unlink (bool): If True (default), the creating process unlinks the shared memory and semaphores once
            its queue object is gone. Other processes must not attach after that.

    The interface follows queue.Queue / multiprocessing.Queue: put(), get(), put_nowait(), get_nowait(),
    qsize(), empty(), full() and shutdown(), raising Empty, Full and ShutDown (which are also queue.Empty,
    queue.Full, TimeoutError and, on Python >= 3.13, queue.ShutDown).

    Example:
        >>> import pickle
        >>> q = DejaQueue(buffer_bytes=1e7)
        >>> pkl = pickle.dumps(q)
        >>> q2 = pickle.loads(pkl)
        >>> q.put({"a": 1, "b": [1,2,3]})
        >>> q2.get()
        {'a': 1, 'b': [1, 2, 3]}
    """

    def put(self, obj, block: bool = True, timeout: float | None = None) -> bool:
        """Put an object into the queue.

        Args:
            obj: Any picklable object (numpy arrays are copied in without an intermediate copy).
            block (bool): If False, raise Full at once if there is no space (timeout is ignored).
            timeout (float | None): Maximum time to wait for space. Default None (wait indefinitely).

        Raises:
            Full: No space in time. ShutDown: The queue was shut down. ValueError: The object is larger than
                the queue's buffer.
        """
        if block is not True:
            timeout = _wait_time(block, timeout, "put")
        if self._state[3]:  # _CLOSED: refuse before pickling (checked again under the lock below)
            raise ShutDown("Queue is shut down.")
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
                    if st[3]:  # _CLOSED
                        raise ShutDown("Queue is shut down.")
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
                        if st[3]:  # _CLOSED while copying: consumers may have stopped already; don't commit
                            raise ShutDown("Queue is shut down.")
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
                raise Full("Timeout waiting for space in queue.")

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

    def get(self, block: bool = True, timeout: float | None = None, *, peek_only: bool = False, callback=None):
        """Get an item from the queue.

        Args:
            block (bool): If False, raise Empty at once if no item is available (timeout is ignored).
            timeout (float | None): Maximum time to wait for an item. Default None (wait indefinitely).
            peek_only (bool): If True, read the item without removing it from the queue. Default False.
            callback (Callable, optional): If provided, a function to be called with the unpickled object
                (zero-copy: its numpy arrays view the queue's shared memory, potentially unsafe!) and whose return
                value is returned. If None (default), the returned object owns all its memory (its numpy arrays are
                copied out of the queue and writable).
                Make sure the callback is fast, since it runs while holding the get_lock, and that it does not keep references
                to the object (or its buffers), since the underlying memory may be overwritten.

        Raises:
            Empty: No item in time. ShutDown: The queue was shut down and is empty.

        Consumers read items one at a time: a get, also with block=False or a timeout, waits while another
        consumer reads an item, including while its callback runs.
        """
        if block is not True:
            if isinstance(timeout, bool) and (block is None or _is_number(block)):
                # dejaq <= 0.9: get(timeout, peek_only); the second argument is a timeout now
                warnings.warn(
                    "DejaQueue.get(): positional (timeout, peek_only) arguments are deprecated; pass "
                    "timeout=... and peek_only=... as keywords",
                    DeprecationWarning,
                    stacklevel=2,
                )
                timeout, peek_only = block, timeout
            else:
                timeout = _wait_time(block, timeout, "get")
        # The item count in the state block (read under the state lock) is the truth about available items.
        # The _items semaphore, with one count per item, serves to wait for items; counts are only taken while
        # holding the get lock, where a consumer that dies is noticed (_GET_CLAIM) and its count restored. Lost
        # or spare counts are harmless: an item is taken whenever one is there, and a count without an item is
        # dropped. So a get with a timeout never waits behind another consumer, and a consumer that dies
        # (even holding the get lock: the OS releases it) cannot block the others.
        deadline = None if timeout is None else (time.monotonic() + max(0.0, float(timeout)))
        items = self._items
        while True:
            self._get_lock.acquire()
            st = self._state  # hot path: literal slot indices
            try:
                if st[7]:  # _GET_CLAIM still set: the previous holder of the lock died holding a count
                    items.release(1)  # restore it (wakes a waiting consumer for the item it left behind)
                st[7] = 1  # set before taking a count: dying in between leaves a spare count, which is harmless
                counted = items.acquire(timeout=0)
                with self._state_lock:
                    nitems = st[2]  # _NITEMS
                    head0 = st[0]  # _HEAD
                if nitems > 0:  # (without a count if it was lost, or is being handed back by a waiting consumer)
                    out = self._get_locked(head0, peek_only, callback, counted)
                    break
                if st[3]:  # _CLOSED (puts are refused under the state lock once it is set)
                    if counted:
                        items.release(1)  # pass shutdown()'s wake-up on to the next waiting consumer
                    raise ShutDown("Queue is shut down and empty.")
                # No item (a count without one is spare, e.g. after purge(): drop it).
            finally:
                st[7] = 0
                self._get_lock.release()
            # Wait for an item without holding the lock. The count only signals that one arrived: hand it back
            # at once and take the item under the lock. (Only a process dying right in between loses a count.)
            rem = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not items.acquire(timeout=rem):
                raise Empty("Timeout waiting for item.")
            items.release(1)
        return out

    def put_nowait(self, obj) -> bool:
        """put(obj, block=False)"""
        return self.put(obj, block=False)

    def get_nowait(self):
        """get(block=False) (waits while another consumer reads an item, see get())"""
        return self.get(block=False)

    def qsize(self) -> int:
        """Number of items in the queue."""
        return self.nitems

    def empty(self) -> bool:
        """True if the queue holds no items (may change at once if other processes use the queue)."""
        return self.nitems == 0

    def full(self) -> bool:
        """True if not even the smallest item fits (a larger item may not fit even when this is False)."""
        return self.bytes_free < _MIN_ITEM_BYTES

    def _get_locked(self, head0: int, peek_only: bool, callback, counted: bool = True):
        """Read the item at head0. Caller holds get_lock, and one count of _items if counted."""
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

        def _private(start, n):
            """Writable private copy of a span (numpy-allocated: arrays built on it are writable and aligned)."""
            out = np.empty(n, dtype=np.uint8)
            dst = memoryview(out)
            s = start % cap
            if s + n <= cap:
                dst[:] = buf[s : s + n]
            else:
                first = cap - s
                dst[:first] = buf[s:cap]
                dst[first:] = buf[0 : n - first]
            dst.release()
            return out

        # header
        K = struct.unpack("<Q", _copy_span(head0, 8))[0]
        lens = struct.unpack(f"<{K}Q", _copy_span(head0 + 8, 8 * K))
        total = 8 + 8 * K + sum(lens)
        start0 = head0 + 8 + 8 * K

        segs, cursor = [], start0
        try:
            for i, n in enumerate(lens):
                s = cursor % cap
                e = s + n
                if callback is None and i > 0:
                    segs.append(_private(cursor, n))  # out-of-band buffers (numpy data): private, writable copies
                elif e <= cap:
                    segs.append(buf[s:e])
                else:
                    segs.append(_copy_span(cursor, n))  # wrapped -> copy
                cursor += n
            obj = _pickle_backend.loads(segs[0], buffers=segs[1:])
            # Without a callback, obj owns all its memory: in-band data is copied by unpickling, out-of-band
            # buffers were copied above. With a callback, its buffers still view the ring.
            out = obj if callback is None else callback(obj)
        except BaseException as e:
            # Never leave a failing item at the head with its count consumed: every later get()
            # would fail on it again. A get that fails while copying/unpickling/in the callback
            # drops the item (as a successful get would); a failed peek, or an interrupt
            # (KeyboardInterrupt, SystemExit), leaves it in the queue.
            segs = obj = None  # release views of the ring before its space is reused
            self._finish_get(head0, total, consume=isinstance(e, Exception) and not peek_only, counted=counted)
            raise
        segs = obj = None
        self._finish_get(head0, total, consume=not peek_only, counted=counted)  # advance only after loads()
        return out

    def _finish_get(self, head0: int, total: int, consume: bool, counted: bool = True) -> None:
        """Caller holds get_lock (and one count of _items if counted). Drop the item at head0, or keep it
        (and put the count back)."""
        if consume:
            st = self._state  # hot path: literal slot indices
            with self._state_lock:
                st[0] = (head0 + total) % self.cap  # _HEAD
                st[2] -= 1  # _NITEMS
                if st[5]:  # _WAITERS
                    self._wake_producers()
        elif counted:
            self._items.release(1)

    def __iter__(self):
        """Yield items until the queue is shut down and empty (or an Ellipsis stop signal arrives, which is
        put back for other consumers)."""
        while True:
            try:
                x = self.get()
            except ShutDown:
                return
            if x is Ellipsis:
                try:
                    self.put(Ellipsis)
                except ShutDown:
                    pass
                return
            yield x

    def _signal_stop(self, n=1):
        """Puts n stop signals into the queue (no-op once it is shut down: consumers stop anyway)."""
        try:
            for _ in range(n):
                self.put(Ellipsis)
        except ShutDown:
            pass


### Legacy DejaQueue: ###

class ByteFIFO:
    """A FIFO buffer (queue) for bytes. The queue is implemented as a ring buffer in shared memory.

    Deprecated (to be removed in dejaq 1.0): use DejaQueue.
    """

    def __init__(self, buffer_bytes=10e6):
        """
        Initializes a ByteFIFO object.

        Args:
            buffer_bytes (int): The size of the buffer in bytes. Defaults to 10 MiB.
        """
        warnings.warn(
            f"{type(self).__name__} is deprecated and will be removed in dejaq 1.0; use DejaQueue",
            DeprecationWarning,
            stacklevel=2 if type(self) is ByteFIFO else 3,
        )

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

    Deprecated (to be removed in dejaq 1.0): use DejaQueue.

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
