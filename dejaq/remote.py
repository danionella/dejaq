"""Run a class instance (Actor) or a function (RemoteFunc) in other processes and call it as if it were local.

Requests go through a shared-memory DejaQueue to the server process(es); each client process has a mailbox
(a DejaQueue of its own) for the replies, which are matched to calls by id. A remote exception is re-raised
with its original type (the remote traceback as its cause), or as RemoteError if it cannot be passed on.
"""
from __future__ import annotations

import copy
import inspect
import itertools
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import warnings
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import util as _mp_util
from types import ModuleType
from typing import Any, Callable, Optional

import cloudpickle
import numpy as np
import psutil

from .queues import DejaQueue, Empty, Full, ShutDown
from .parallel import _WorkerException, _exit_with_parent, _stop_processes

logger = logging.getLogger(__name__)

__all__ = ["Actor", "ActorDecorator", "RemoteFunc", "RemoteError", "Future"]

_POLL = 0.5  # seconds between liveness checks while waiting (only then: no cost per call)
_REPLY_WAIT = 30.0  # how long a server waits for space in a client's full mailbox before dropping the reply
_KEEP_MAILBOXES = 16  # mailboxes a server keeps open (one per client process)
_ids = itertools.count()  # call ids: unique per process, hence per mailbox


class RemoteError(RuntimeError):
    """The remote process died or was closed, or raised an exception that could not be passed on."""


@dataclass
class _Req:
    call_id: int
    mailbox: Optional[str]  # name of the caller's mailbox, or None: no reply wanted
    kind: str  # call, apply, getattr, setattr, resolve, dir, ping, shutdown
    name: str
    args: tuple
    kwargs: dict
    deepcopy: bool = True


@dataclass
class _Rep:
    call_id: int
    ok: bool
    payload: Any  # the result, or a _WorkerException


# ============================== servers ==============================


class _Replier:
    """Server side: sends replies to the clients' mailboxes, keeping the most recently used ones open."""

    def __init__(self) -> None:
        self._open: "OrderedDict[str, DejaQueue]" = OrderedDict()

    def __call__(self, cid: int, mailbox: Optional[str], ok: bool, payload: Any) -> None:
        if mailbox is None:
            return
        try:
            q = self._open.get(mailbox)
            if q is None:
                q = self._open[mailbox] = DejaQueue(name=mailbox, create=False)
                if len(self._open) > _KEEP_MAILBOXES:
                    self._open.popitem(last=False)[1].close()
            else:
                self._open.move_to_end(mailbox)
            try:
                q.put(_Rep(cid, ok, payload), timeout=_REPLY_WAIT)
            except (Full, ShutDown):
                raise
            except Exception as e:  # e.g. the result cannot be pickled: send the error instead
                q.put(_Rep(cid, False, _WorkerException(e, RemoteError)), timeout=_REPLY_WAIT)
        except Exception as e:  # the client is gone, or does not read its mailbox
            logger.warning(f"reply to call {cid} dropped: {e!r}")


def _static(obj, name: str):
    try:
        return inspect.getattr_static(obj, name)
    except AttributeError:
        return None


def _own(obj, name: str):
    """obj.name if set on the instance or its class (fast, and without __getattr__), else None."""
    d = getattr(obj, "__dict__", None)
    return d[name] if d is not None and name in d else getattr(type(obj), name, None)


def _resolve(obj, name: str) -> dict:
    """Whether obj has `name` and whether it is a method (without running properties)."""
    v = _static(obj, name)
    if v is None:
        try:
            v = getattr(obj, name)  # a dynamic attribute (__getattr__)
        except AttributeError:
            return {"exists": False}
    if isinstance(v, (staticmethod, classmethod)):
        v = v.__func__
    method = callable(v) and not isinstance(v, property)
    try:
        sig = inspect.signature(v) if method else None
    except (TypeError, ValueError):  # e.g. some builtins
        sig = None
    return {"exists": True, "callable": method, "signature": sig}


def _actor_server(parent: int, pkl: bytes) -> None:
    _exit_with_parent(parent)
    cls, args, kwargs, req_name = cloudpickle.loads(pkl)
    req, reply = DejaQueue(name=req_name, create=False), _Replier()
    try:
        obj, failed = (cls if isinstance(cls, ModuleType) else cls(*args, **kwargs)), None
    except BaseException as e:  # every call raises the constructor's exception
        obj, failed = None, _WorkerException(e, RemoteError)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def handle(kind, name, args, kwargs, deep):
        if failed is not None:
            raise failed.exc
        if kind == "call":
            args, kwargs = copy.deepcopy((args, kwargs)) if deep else (args, kwargs)
            return getattr(obj, name)(*args, **kwargs)
        if kind == "getattr":
            return getattr(obj, name)
        if kind == "setattr":
            setattr(obj, name, copy.deepcopy(kwargs["value"]) if deep else kwargs["value"])
            return True
        if kind == "resolve":
            return _resolve(obj, name)
        if kind == "dir":
            return list(dict.fromkeys(dir(obj)))
        if kind == "ping":
            return time.time()
        raise ValueError(f"unknown request {kind!r}")

    def dispatch(msg: _Req) -> bool:
        # Runs inside req.get(callback=...): with deepcopy=False, the arguments' arrays view the request queue,
        # so the reply is sent before that memory is reused (a result that views an argument is then safe).
        if msg.kind == "shutdown":
            return False
        bases = [] if msg.deepcopy else [a.base for a in (*msg.args, *msg.kwargs.values())
                                         if isinstance(a, np.ndarray) and a.base is not None]
        refs = [sys.getrefcount(b) for b in bases]
        try:
            reply(msg.call_id, msg.mailbox, True, handle(msg.kind, msg.name, msg.args, msg.kwargs, msg.deepcopy))
        except BaseException as e:
            reply(msg.call_id, msg.mailbox, False, _WorkerException(e, RemoteError))
        if any(sys.getrefcount(b) > n for b, n in zip(bases, refs)):  # (views of an argument count, too)
            warnings.warn(f"{msg.name!r} kept a reference to an argument passed with deepcopy=False: its "
                          "memory is reused once the call returns (copy what you need to keep)", RuntimeWarning)
        return True

    while not stop.is_set():
        try:
            if not req.get(timeout=_own(obj, "_idle_timeout"), callback=dispatch):
                break
        except Empty:
            if _own(obj, "_idle_function"):
                getattr(obj, obj._idle_function)()
        except Exception as e:  # a request that cannot be unpickled: its caller is unknown
            if not stop.is_set():  # (else a signal interrupted the wait)
                logger.error(f"request dropped: {e!r}")
    if obj is not None and hasattr(obj, "close"):
        try:
            obj.close()
        except Exception as e:
            logger.error(f"close() raised {e!r}")


def _func_worker(parent: int, fn_pkl: bytes, req_name: str) -> None:
    _exit_with_parent(parent)
    req, reply = DejaQueue(name=req_name, create=False), _Replier()
    try:
        fn, failed = cloudpickle.loads(fn_pkl), None
    except BaseException as e:
        fn, failed = None, _WorkerException(e, RemoteError)
    while True:
        try:
            msg = req.get()
        except Exception as e:  # a request that cannot be unpickled: its caller is unknown
            logger.error(f"request dropped: {e!r}")
            continue
        if msg.kind == "shutdown":
            break
        try:
            if failed is not None:
                raise failed.exc
            reply(msg.call_id, msg.mailbox, True, fn(*msg.args, **msg.kwargs))
        except BaseException as e:
            reply(msg.call_id, msg.mailbox, False, _WorkerException(e, RemoteError))


# ============================== clients ==============================


class _Mailbox:
    """A client process's replies, matched to calls by id. Thread-safe: one thread reads at a time."""

    def __init__(self, buffer_bytes: int) -> None:
        self.q = DejaQueue(buffer_bytes)
        self.name = self.q._base
        self._got: dict = {}
        self._dropped: set = set()  # calls given up on (timeout): their replies are discarded
        self._cond = threading.Condition()
        self._reading = False

    def _read(self, timeout: float) -> bool:
        """Store one reply (if one comes within timeout); False if none came. Caller holds _cond."""
        if self._reading:  # another thread reads: it stores our reply, too
            self._cond.wait(timeout)
            return False
        self._reading = True
        self._cond.release()
        try:
            rep = self.q.get(timeout=timeout)
        except Empty:
            rep = None
        finally:
            self._cond.acquire()
            self._reading = False
            self._cond.notify_all()
        if rep is None:
            return False
        if rep.call_id in self._dropped:
            self._dropped.discard(rep.call_id)
        else:
            self._got[rep.call_id] = rep
        return True

    def wait(self, cid: int, deadline: Optional[float], alive: Callable[[], bool]) -> _Rep:
        with self._cond:
            while cid not in self._got:
                rem = _POLL if deadline is None else min(_POLL, deadline - time.monotonic())
                if rem <= 0:
                    self._dropped.add(cid)
                    raise TimeoutError(f"no reply within the timeout (call {cid})")
                t0 = time.monotonic()
                self._read(rem)
                if cid not in self._got and time.monotonic() - t0 >= _POLL and not alive():
                    raise RemoteError("the remote process has exited (it died or was closed)")
            return self._got.pop(cid)

    def pump(self) -> None:
        """Store the replies that have arrived (e.g. while a request waits for space: the server may wait for us)."""
        with self._cond:
            while self._read(0):
                pass


class Future:
    """The result of a remote call that was not waited for."""

    def __init__(self, client: "_Client", cid: int) -> None:
        self._client, self._id = client, cid

    def result(self, timeout: Optional[float] = None):
        if not hasattr(self, "_rep"):
            deadline = None if timeout is None else time.monotonic() + timeout
            self._rep = self._client._mbox.wait(self._id, deadline, self._client._alive)
        return _unwrap(self._rep)


def _unwrap(rep: _Rep):
    if rep.ok:
        return rep.payload
    rep.payload.reraise()


def _stop_servers(procs, req: DejaQueue, timeout: float = 1.0) -> None:
    for _ in procs:
        try:
            req.put(_Req(-1, None, "shutdown", "", (), {}), timeout=0)
        except Exception:
            break  # (stopped below anyway)
    _stop_processes(procs, grace=timeout)


_mbox_lock = threading.Lock()
_mailboxes: "weakref.WeakValueDictionary[tuple, _Mailbox]" = weakref.WeakValueDictionary()  # of this process


class _Client:
    """Shared by Actor and RemoteFunc: requests, the mailbox of this process, liveness, stopping."""

    def _start(self, procs, req: DejaQueue, buffer_bytes: int) -> None:
        for p in procs:
            p.start()
        self._req, self._procs, self._buffer_bytes = req, procs, int(buffer_bytes)
        self._servers = [(p.pid, psutil.Process(p.pid).create_time()) for p in procs]
        self._closed = False
        self._finalizer = _mp_util.Finalize(self, _stop_servers, args=(procs, req), exitpriority=10)

    @property
    def _mbox(self) -> _Mailbox:
        """This process's mailbox for this client (created on first use: also in processes a client was pickled
        or forked to; shared by all copies of the client in a process)."""
        mb = self.__dict__.get("_mb")
        if mb is None or mb[0] != os.getpid():
            key = (os.getpid(), self._req._base)
            with _mbox_lock:
                box = _mailboxes.get(key)
                if box is None:
                    _mailboxes[key] = box = _Mailbox(self._buffer_bytes)
            self.__dict__["_mb"] = mb = (os.getpid(), box)
        return mb[1]

    def _alive(self) -> bool:
        """Whether all server processes still run (also works in processes the client was pickled to)."""
        for pid, started in self._servers:
            try:
                p = psutil.Process(pid)
                if p.create_time() != started or p.status() == psutil.STATUS_ZOMBIE:
                    return False
            except psutil.Error:
                return False
        return True

    def _request(self, kind, name, args, kwargs, *, reply=True, deepcopy=True, deadline=None) -> int:
        if self._closed:
            raise RemoteError("closed")
        cid = next(_ids)
        msg = _Req(cid, self._mbox.name if reply else None, kind, name, args, kwargs, deepcopy)
        while True:
            rem = 0.05 if deadline is None else min(0.05, deadline - time.monotonic())
            try:
                self._req.put(msg, timeout=max(rem, 0))
                return cid
            except Full:  # the server may be waiting to deliver replies to us: read them meanwhile
                if "_mb" in self.__dict__:
                    self._mbox.pump()
                if not self._alive():
                    raise RemoteError("the remote process has exited (it died or was closed)") from None
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("the request queue stayed full until the timeout") from None

    def _call(self, kind, name="", args=(), kwargs=None, *, timeout=None, deepcopy=True):
        deadline = None if timeout is None else time.monotonic() + timeout
        cid = self._request(kind, name, args, kwargs or {}, deepcopy=deepcopy, deadline=deadline)
        return _unwrap(self._mbox.wait(cid, deadline, self._alive))

    def close(self, timeout: float = 1.0) -> None:
        """Stop the server process(es): ask them to end, and terminate them after `timeout` seconds."""
        if self.__dict__.get("_finalizer") is not None and not self._closed:
            self._closed = True
            self._finalizer.cancel()
            _stop_servers(self._procs, self._req, timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        state = dict(self.__dict__)
        for k in ("_procs", "_finalizer", "_mb"):  # a copy never stops the servers, and gets its own mailbox
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class _RemoteMethod:
    """A method of an Actor: call it like the method itself."""

    def __init__(self, actor: "Actor", name: str, signature=None) -> None:
        self._actor, self._name = actor, name
        if signature is not None:
            self.__signature__ = signature

    def __call__(self, *args, timeout=None, noreply=False, deepcopy=True, **kwargs):
        """Call the remote method and return its result.

        Args:
          timeout: seconds to wait for the result (TimeoutError); default: no limit.
          noreply: if True, return None at once: the result and any exception are discarded.
          deepcopy: if False, numpy array arguments view shared memory instead of being copied (zero-copy): they
            are valid only during the call (copy what you need to keep; returning them is fine).
        """
        if noreply:
            self._actor._request("call", self._name, args, kwargs, reply=False, deepcopy=deepcopy)
            return None
        return self._actor._call("call", self._name, args, kwargs, timeout=timeout, deepcopy=deepcopy)

    def submit(self, *args, deepcopy=True, **kwargs) -> Future:
        """Call the remote method without waiting: returns a Future."""
        return Future(self._actor, self._actor._request("call", self._name, args, kwargs, deepcopy=deepcopy))

    def send(self, *args, **kwargs) -> None:
        """Call without waiting for (or receiving) the result (noreply=True)."""
        self(*args, noreply=True, **kwargs)

    def __repr__(self) -> str:
        return f"<remote method {self._name} of {self._actor!r}>"


_REMOTE_PRIVATE = ("__call__", "_idle_function", "_idle_timeout")  # private names of the remote object


class Actor(_Client):
    """Run an instance of `cls` in a process of its own; its methods and attributes are used remotely.

      • calls: `a.method(x)`; without waiting: `a.method_async(x)` or `a.method.submit(x)` (Future), or
        `a.method(x, noreply=True)`; with `timeout=...`; zero-copy array arguments with `deepcopy=False`
      • attributes: `a.value`, `a.value = 1` (properties run remotely)
      • the actor stops at `close()`, at the end of a `with` block, when the Actor is garbage-collected, or
        at exit. An Actor can be passed to other processes (each gets a mailbox of its own); a copy never
        stops the actor.
      • the remote object may define `_idle_timeout` (seconds) and `_idle_function` (name of a method that is
        called when no request came within that time)
      • a request the actor cannot unpickle (e.g. an argument of a class defined only in a notebook) is
        dropped with an error in the actor's log: its call waits until its timeout

    Args:
      cls: the class (local and nested classes work too), or a module
      *args, **kwargs: constructor arguments
      buffer_bytes: size of the request queue and of each mailbox (default 10 MB)
      start_method: multiprocessing start method (default 'spawn')
    """

    def __init__(self, cls, *args, buffer_bytes: int = int(10e6), start_method: str = "spawn", **kwargs) -> None:
        req = DejaQueue(buffer_bytes)
        pkl = cloudpickle.dumps((cls, args, kwargs, req._base))
        self._start([mp.get_context(start_method).Process(target=_actor_server, args=(os.getpid(), pkl))],
                    req, buffer_bytes)
        self._cls_name = getattr(getattr(cls, "__wrapped__", cls), "__name__", repr(cls))
        self._methods: dict = {}  # name -> signature of methods resolved so far

    def __getattr__(self, name: str):
        if name.startswith("_") and name not in _REMOTE_PRIVATE:
            raise AttributeError(name)
        if name.endswith("_async"):
            return _RemoteMethod(self, name[:-6]).submit
        if name not in self._methods:
            meta = self._call("resolve", name)
            if not meta["exists"]:
                raise AttributeError(name)
            if not meta["callable"]:
                return self._call("getattr", name)  # (not cached: the value may change)
            self._methods[name] = meta["signature"]
        return _RemoteMethod(self, name, self._methods[name])

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") and name not in _REMOTE_PRIVATE:
            object.__setattr__(self, name, value)
        else:
            self._methods.pop(name, None)
            self._call("setattr", name, (), {"value": value})

    def __dir__(self):
        try:
            return sorted(set(super().__dir__()) | set(self._call("dir", timeout=1.0)))
        except Exception:  # (tab completion must not fail)
            return super().__dir__()

    def ping(self) -> dict:
        """Round-trip timings in ms."""
        t0 = time.time()
        t1 = self._call("ping")
        t2 = time.time()
        return {"out_ms": (t1 - t0) * 1e3, "back_ms": (t2 - t1) * 1e3, "total_ms": (t2 - t0) * 1e3}

    def __repr__(self) -> str:
        return f"<Actor (PID {self._servers[0][0]}) for an instance of {self._cls_name}>"


class RemoteFunc(_Client):
    """Run a function in a pool of worker processes.

    Example:
        >>> rf = RemoteFunc(square, workers=4)
        >>> rf(12)                            # 144
        >>> rf.map(range(5))                  # [0, 1, 4, 9, 16]
        >>> rf.submit(7).result()             # 49
        >>> rf.close()

    Args:
      fn: the function (local functions and lambdas work too)
      workers: number of worker processes
      buffer_bytes: size of the request queue and of each mailbox
      start_method: multiprocessing start method (default 'spawn')
    """

    def __init__(self, fn: Callable, workers: int = 1, buffer_bytes: int = 8_000_000, start_method: str = "spawn"):
        req = DejaQueue(buffer_bytes)
        ctx, pkl = mp.get_context(start_method), cloudpickle.dumps(fn)
        procs = [ctx.Process(target=_func_worker, args=(os.getpid(), pkl, req._base)) for _ in range(int(workers))]
        self._start(procs, req, buffer_bytes)

    def __call__(self, *args, timeout: Optional[float] = None, **kwargs):
        """fn(*args, **kwargs), computed by a worker."""
        return self._call("apply", "", args, kwargs, timeout=timeout)

    def submit(self, *args, **kwargs) -> Future:
        """fn(*args, **kwargs) without waiting: returns a Future."""
        return Future(self, self._request("apply", "", args, kwargs))

    def map(self, iterable, *, timeout: Optional[float] = None) -> list:
        """[fn(x) for x in iterable], computed by the workers (each result waited for up to `timeout` seconds)."""
        return [f.result(timeout) for f in [self.submit(x) for x in iterable]]


def ActorDecorator(cls=None, **decorator_kwargs):
    """Make a class create Actors: `@ActorDecorator` or `@ActorDecorator(buffer_bytes=1e6)`; then `Counter(10)`
    returns an Actor running `Counter(10)`. Keyword arguments of the decorator are passed to Actor."""
    if cls is None:
        return lambda c: ActorDecorator(c, **decorator_kwargs)

    def WrappedActor(*args, **kwargs):
        return Actor(cls, *args, **{**kwargs, **decorator_kwargs})

    return WrappedActor
