from __future__ import annotations

import logging
import os
import time
import uuid
import traceback
import psutil
import inspect
import multiprocessing as mp
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, Tuple, List
from types import ModuleType

import cloudpickle

from dejaq.queues import DejaQueue


@dataclass
class _Req:
    """Request envelope.

    Attributes:
        call_id: Correlation id for matching replies.
        reply: Name/base of the caller's mailbox queue, or None to suppress replies.
        kind: One of {"call","apply","ping","shutdown","getattr","setattr","resolve","dir"}.
        name: Method/attribute name (actor) or unused (remote func).
        args: Positional arguments.
        kwargs: Keyword arguments (for setattr: {"value": ...}).
    """

    call_id: str
    reply: Optional[str]  # <-- allow None to mean "no reply requested"
    kind: str
    name: str
    args: tuple
    kwargs: dict
    cloudpickle_result: bool = False  # <-- whether to serialize the result with cloudpickle


@dataclass
class _Rep:
    """Reply envelope.

    Attributes:
        call_id: Correlation id (matches the request).
        ok: True if the call succeeded; False if it raised.
        payload: Result object (if ok), else a tuple (etype, eargs, tb_str).
    """

    call_id: str
    ok: bool
    payload: Any


class RemoteError(RuntimeError):
    """Exception raised locally for errors that occurred in a remote worker/actor."""

    def __init__(self, etype: str, eargs: Tuple[Any, ...], remote_tb: str):
        super().__init__(f"{etype}{eargs}\n-- Remote traceback --\n{remote_tb}")
        self.remote_type = etype
        self.remote_args = eargs
        self.remote_traceback = remote_tb


# ============================== Server/worker loops ==============================


def _actor_server(pkl: bytes) -> None:
    """Actor loop: instantiates `cls` and services _Req from a request queue.

    The server replies to the mailbox specified by each request's `reply`
    (skipped entirely if `reply` is None). Designed to be module-level for Windows 'spawn'.
    """
    cls, actor_args, actor_kwargs, req_name = cloudpickle.loads(pkl)
    req = DejaQueue(name=req_name, create=False)
    if isinstance(cls, ModuleType):
        obj = cls
    else:
        obj = cls(*actor_args, **actor_kwargs)

    rep_cache: Dict[str, DejaQueue] = {}

    def repq(name: str) -> DejaQueue:
        q = rep_cache.get(name)
        if q is None:
            q = DejaQueue(name=name, create=False)
            rep_cache[name] = q
        return q

    def _maybe_reply(msg: _Req, rep: _Rep) -> None:
        # Only reply if requested (fire-and-forget otherwise)
        if msg.reply is not None:
            repq(msg.reply).put(rep)

    def _dir_payload() -> Dict[str, List[str]]:
        """List names/methods/attrs without triggering property getters."""
        names = list(dict.fromkeys(dir(obj)))  # stable & unique
        methods: List[str] = []
        attrs: List[str] = []
        for n in names:
            try:
                v = inspect.getattr_static(obj, n)
            except Exception:
                continue
            # unwrap common descriptors
            if isinstance(v, (staticmethod, classmethod)):
                v = v.__func__
            if isinstance(v, property):
                attrs.append(n)
            elif callable(v):
                methods.append(n)
            else:
                attrs.append(n)
        return {"names": names, "methods": methods, "attrs": attrs}

    while True:
        msg: _Req = req.get()  # blocking
        if msg.kind == "shutdown":
            _maybe_reply(msg, _Rep(msg.call_id, True, None))
            break
        try:
            if msg.kind == "ping":
                out = time.time()
                _maybe_reply(msg, _Rep(msg.call_id, True, out))
            elif msg.kind == "call":
                out = getattr(obj, msg.name)(*msg.args, **msg.kwargs)
                if msg.cloudpickle_result:
                    out = cloudpickle.dumps(out)
                _maybe_reply(msg, _Rep(msg.call_id, True, out))
            elif msg.kind == "getattr":
                out = getattr(obj, msg.name)
                _maybe_reply(msg, _Rep(msg.call_id, True, out))
            elif msg.kind == "setattr":
                setattr(obj, msg.name, msg.kwargs.get("value"))
                _maybe_reply(msg, _Rep(msg.call_id, True, True))
            elif msg.kind == "resolve":  # classify *without* invoking descriptors/properties
                try:
                    v = inspect.getattr_static(obj, msg.name)
                except AttributeError:
                    try:  # Fallback to the unwrapped target (common for proxies)
                        v = inspect.getattr_static(inspect.unwrap(obj), msg.name)
                    except AttributeError:
                        out = {"exists": False, "callable": False}
                        _maybe_reply(msg, _Rep(msg.call_id, True, out))
                        continue
                if isinstance(v, (staticmethod, classmethod)):
                    v = v.__func__
                is_prop = isinstance(v, property)
                out = {
                    "exists": True,
                    "callable": (callable(v) and not is_prop),
                    "signature": inspect.signature(v) if callable(v) else None,
                }
                _maybe_reply(msg, _Rep(msg.call_id, True, out))
            elif msg.kind == "dir":
                out = _dir_payload()
                _maybe_reply(msg, _Rep(msg.call_id, True, out))
            else:
                raise ValueError(f"unknown kind {msg.kind!r}")
        except BaseException as e:
            if msg.reply is not None:  # Only attempt to send the error back if a reply was requested
                _maybe_reply(msg, _Rep(msg.call_id, False, (type(e).__name__, e.args, traceback.format_exc())))
            else:  # else: swallow — caller explicitly opted out of replies
                logging.error(f"Actor server caught: {type(e).__name__}{e.args}\n{traceback.format_exc()}")


def _func_worker(fn_ser: Tuple[str, str] | Callable, req_name: str) -> None:
    """Function worker loop: applies a function for 'apply' requests.

    Args:
      fn_ser: Either a callable or a (module, name) pair for lazy import in the worker.
      req_name: Name/base of the shared request queue.
    """
    if callable(fn_ser):
        fn = fn_ser
    else:
        mod, name = fn_ser
        fn = getattr(__import__(mod, fromlist=[name]), name)

    req = DejaQueue(name=req_name, create=False)
    rep_cache: Dict[str, DejaQueue] = {}

    def repq(name: str) -> DejaQueue:
        q = rep_cache.get(name)
        if q is None:
            q = DejaQueue(name=name, create=False)
            rep_cache[name] = q
        return q

    def _maybe_reply(msg: _Req, rep: _Rep) -> None:
        if msg.reply is not None:
            repq(msg.reply).put(rep)

    while True:
        msg: _Req = req.get()
        if msg.kind == "shutdown":
            _maybe_reply(msg, _Rep(msg.call_id, True, None))
            break
        try:
            if msg.kind != "apply":
                raise ValueError(f"unknown kind {msg.kind!r}")
            out = fn(*msg.args, **msg.kwargs)
            _maybe_reply(msg, _Rep(msg.call_id, True, out))
        except BaseException as e:
            if msg.reply is not None:
                _maybe_reply(msg, _Rep(msg.call_id, False, (type(e).__name__, e.args, traceback.format_exc())))


# ============================== Client-side demux & futures ==============================


class _Mailbox:
    """Demultiplex replies by `call_id` for a single client process.

    Keeps a small buffer of out-of-order replies so multiple Futures can be awaited
    concurrently. Each *process* gets its own mailbox queue; servers reply to the
    specific mailbox requested → no cross-process races.

    Args:
      rep_q: The reply/mailbox queue owned by this process.
    """

    def __init__(self, rep_q: DejaQueue) -> None:
        self.q = rep_q
        self._buf: Dict[str, _Rep] = {}

    def wait(self, call_id: str, timeout: Optional[float]) -> _Rep:
        """Wait for the reply with matching `call_id`."""
        if call_id in self._buf:
            return self._buf.pop(call_id)
        deadline = None if timeout is None else (time.time() + timeout)
        while True:
            rem = None if deadline is None else max(0.0, deadline - time.time())
            rep: _Rep = self.q.get(timeout=rem)  # raises TimeoutError on deadline
            if rep.call_id == call_id:
                return rep
            self._buf[rep.call_id] = rep

    def msg_arrived(self, call_id: str) -> bool:
        """Check if a message with `call_id` has arrived (non-blocking)."""
        if call_id in self._buf:
            return True
        try:
            while True:
                rep: _Rep = self.q.get(timeout=0.0)
                if rep.call_id == call_id:
                    return True
                self._buf[rep.call_id] = rep
        except TimeoutError:
            return False


class Future:
    """Handle for an outstanding remote call."""

    def __init__(self, mbox: _Mailbox, call_id: str) -> None:
        self._mbox, self._id = mbox, call_id

    def result(self, timeout: Optional[float] = None):
        """Return the remote result or raise RemoteError."""
        rep = self._mbox.wait(self._id, timeout)
        if rep.ok:
            return rep.payload
        et, ea, tb = rep.payload
        raise RemoteError(et, ea, tb)


# ============================== Public API: Actor & RemoteFunc ==============================


class _RemoteMethod:
    """Lightweight callable proxy for a remote method."""

    def __init__(self, actor_ref: "Actor", name: str, signature: Optional[inspect.Signature] = None) -> None:
        self._actor = weakref.proxy(actor_ref)
        self._name = name
        self._signature = signature

    def __call__(
        self,
        *args,
        timeout: Optional[float] = None,
        noreply: bool = False,
        _use_cloudpickle=False,
        _ensure_open=False,
        **kwargs,
    ):
        """Invoke the remote method.

        Args:
          *args, **kwargs: Forwarded to the remote method.
          timeout: Seconds to wait for the result; ignored if `noreply` is True.
          noreply: If True, fire-and-forget — do not request or wait for a reply.

        Returns:
          The remote return value (when noreply=False). Returns None when noreply=True.

        Notes:
          • When `noreply=True`, remote exceptions are not propagated.
          • Use this for high-throughput paths where acknowledgement is unnecessary.
        """
        if _ensure_open:
            self._actor._ensure_open()
        cid = self._actor._send(
            "call",
            self._name,
            args,
            kwargs,
            expect_reply=not noreply,
            _use_cloudpickle=_use_cloudpickle,
        )
        if noreply:
            return None
        rep = self._actor._mbox.wait(cid, timeout)
        if rep.ok:
            return rep.payload if not _use_cloudpickle else cloudpickle.loads(rep.payload)
        et, ea, tb = rep.payload
        raise RemoteError(et, ea, tb)

    def send(self, *args, **kwargs) -> None:
        """Convenience: fire-and-forget call (equivalent to noreply=True)."""
        self(*args, noreply=True, **kwargs)

    def __repr__(self) -> str:
        return f"<RemoteMethod {self._name}>"

    __doc__ = "Remote method; call to execute in the actor process."


class Actor:
    """Run a class instance in a separate process and call its methods/attributes remotely.

    Supports:
      • Remote method calls: `a.method(x)`, `a.method_async(x)`, `a.method(..., noreply=True)`
      • Jupyter tab completion: `__dir__` merges local + remote names

    Args:
      cls: Class to run in the actor process (nested/local supported via cloudpickle).
      *args: Positional args for the class constructor.
      buffer_bytes: Size of each queue (request/mailbox) in bytes. Default 10e6 (10 MiB).
      start_method: Multiprocessing start method (default 'spawn' for portability).
      **kwargs: Keyword args for the class constructor.
    """

    # --- construction ---
    def __init__(self, cls: type, *args, buffer_bytes: int = int(10e6), start_method: str = "spawn", **kwargs) -> None:
        base = f"act-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._rep = DejaQueue(buffer_bytes=buffer_bytes, name=base + "_mb", create=True)  # mailbox
        self._mbox = _Mailbox(self._rep)
        self._req = DejaQueue(buffer_bytes=buffer_bytes, name=base + "_req", create=True)  # requests
        ctx = mp.get_context(start_method)
        pkl = cloudpickle.dumps((cls, args, kwargs, self._req._base))
        logging.info(f"args: {args}, kwargs: {kwargs}, base: {base}")
        ps = [ctx.Process(target=_actor_server, args=(pkl,)) for _ in range(1)]
        [p.start() for p in ps]
        logging.info(f"Actor: started process with PID {[p.pid for p in ps]}")
        self._proc_meta = [{"pid": p.pid, "create_time": psutil.Process(p.pid).create_time()} for p in ps]
        self._cache = {}  # Cache for resolved remote methods/attributes

    # --- internals ---
    def is_proc_alive(self, _proc_meta) -> bool:
        try:
            pr = psutil.Process(_proc_meta["pid"])
            return (
                pr.create_time() == _proc_meta["create_time"]
                and pr.is_running()
            )
        except psutil.NoSuchProcess:
            return False

    def _ensure_open(self) -> None:
        """Raise RuntimeError if the actor is closed or its process is dead."""
        # TODO: decide whether to keep abillity to have multiple PIDs, or simplify
        if not any([self.is_proc_alive(_proc_meta) for _proc_meta in self._proc_meta]):
            raise RuntimeError("Actor process has exited")

    def _send(
        self,
        kind: str,
        name: str,
        args: tuple,
        kwargs: dict,
        *,
        expect_reply: bool = True,
        _use_cloudpickle: bool = False,
    ) -> str:
        cid = uuid.uuid4().hex
        reply = self._rep._base if expect_reply else None
        self._req.put(_Req(cid, reply, kind, name, args, kwargs, cloudpickle_result=_use_cloudpickle))
        return cid

    # --- dynamic attribute/method resolution ---
    def __getattr__(self, name: str):
        """Dynamic proxy resolution.

        Behavior:
          • If `name` ends with `_async`, return an async method proxy.
          • Classify via 'resolve' (safe, no side effects). If callable → method proxy.
          • Otherwise fetch value via 'getattr' and return it.

        Raises:
          AttributeError: If the remote object has no such attribute.
          RemoteError: If the remote getattr/resolve raised.
        """
        if name.startswith("_") and name != "__call__":
            raise AttributeError(name)

        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # async sugar
        if name.endswith("_async"):
            real = name[:-6]

            def _async(*args, **kwargs):
                cid = self._send("call", real, args, kwargs, expect_reply=True)
                return Future(self._mbox, cid)

            self._cache[name] = _async
            return _async

        # classify safely via resolve
        cid = self._send("resolve", name, (), {}, expect_reply=True)
        rep = self._mbox.wait(cid, timeout=2.0)
        if not rep.ok:
            et, ea, tb = rep.payload
            raise RemoteError(et, ea, tb)
        meta = rep.payload  # {"exists": bool, "callable": bool}
        if not meta.get("exists", False):
            raise AttributeError(name)
        if meta.get("callable", False):  # if True
            method = _RemoteMethod(self, name, signature=meta.get("signature", None))
            self._cache[name] = method
            return method

        # non-callable attribute: fetch its value (this may execute properties by design)
        cid = self._send("getattr", name, (), {}, expect_reply=True)
        rep = self._mbox.wait(cid, timeout=2.0)
        if not rep.ok:
            et, ea, tb = rep.payload
            raise RemoteError(et, ea, tb)
        value = rep.payload
        self._cache[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign to remote attributes for public names; keep locals for private ones.

        Public attribute assignment (no leading underscore) is sent to the remote actor:
            a.foo = 123     # -> setattr on remote object

        Private names (starting with "_") are set locally on the proxy.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        cid = self._send("setattr", name, (), {"value": value}, expect_reply=True)
        rep = self._mbox.wait(cid, timeout=2.0)
        if not rep.ok:
            et, ea, tb = rep.payload
            raise RemoteError(et, ea, tb)

    def _dir(self) -> Dict[str, List[str]]:
        """Return a fresh listing of remote names/methods/attrs (no client cache)."""
        cid = self._send("dir", "", (), {}, expect_reply=True)
        rep = self._mbox.wait(cid, timeout=1.0)  # short timeout to keep completion snappy
        if not rep.ok:
            et, ea, tb = rep.payload
            raise RemoteError(et, ea, tb)
        return rep.payload

    def __dir__(self) -> List[str]:
        """Merge local names with a fresh remote dir() result every time (for Jupyter)."""
        local = set(super().__dir__())
        try:
            names = set(self._dir().get("names", []))
            return sorted(local | names)
        except Exception:
            # Timeout/errors during completion should not explode user typing
            return sorted(local)

    def ping(self, _ensure_open=False) -> Dict[str, float]:
        """Liveness/latency probe. Returns timings in seconds."""
        t0 = time.time()
        if _ensure_open:
            self._ensure_open()
        cid = self._send("ping", "", (), {}, expect_reply=True)
        t1 = self._mbox.wait(cid, 2.0).payload
        t2 = time.time()
        return {"out_ms": (t1 - t0) * 1000, "back_ms": (t2 - t1) * 1000, "total_ms": (t2 - t0) * 1000}

    def close(self, timeout: float = 2.0) -> None:
        """Gracefully stop the actor process."""
        try:
            cid = self._send("shutdown", "", (), {}, expect_reply=True)
            _ = self._mbox.wait(cid, timeout)
        except Exception:
            logging.warning("Actor.close: graceful shutdown failed!")
        # self._p.join(timeout)
        # if self._p.is_alive():
        #     self._p.terminate()
        #     self._p.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class RemoteFunc:
    """Run a function in N worker processes (stateless pool).

    Each call is a request on a shared work queue; each *client process* has its
    own mailbox queue for replies (no cross-process demux races).

    Example:
        >>> def f(x): return x * x
        >>> rf = RemoteFunc(f, workers=4)
        >>> rf(12)               # 144
        >>> rf.map(range(5))     # [0, 1, 4, 9, 16]
        >>> fut = rf.submit(7); fut.result()  # 49
        >>> rf.close()

    Args:
      fn: Callable to execute in workers (importable top-level preferred).
      workers: Number of worker processes.
      buffer_bytes: Size of each queue (request/mailbox) in bytes.
      start_method: Multiprocessing start method (default 'spawn').
    """

    def __init__(
        self, fn: Callable, workers: int = 1, buffer_bytes: int = 8_000_000, start_method: str = "spawn"
    ) -> None:
        base = f"rf-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._rep = DejaQueue(buffer_bytes=buffer_bytes, name=base + "_mb", create=True)  # mailbox
        self._mbox = _Mailbox(self._rep)
        self._req = DejaQueue(buffer_bytes=buffer_bytes, name=base + "_req", create=True)  # work queue
        ctx = mp.get_context(start_method)

        # Prefer (module, name) reference for spawn-friendliness; fall back to pickled callable.
        try:
            mod, name = fn.__module__, fn.__name__
            getattr(__import__(mod, fromlist=[name]), name)  # validate importability
            fn_ref: Tuple[str, str] | Callable = (mod, name)
        except Exception:
            fn_ref = fn

        self._ps = [ctx.Process(target=_func_worker, args=(fn_ref, self._req._base)) for _ in range(int(workers))]
        for p in self._ps:
            p.start()

    def __call__(self, *args, timeout: Optional[float] = None, **kwargs):
        """Apply fn(*args, **kwargs) and block for the result."""
        cid = uuid.uuid4().hex
        self._req.put(_Req(cid, self._rep._base, "apply", "", args, kwargs))
        rep = self._mbox.wait(cid, timeout)
        if rep.ok:
            return rep.payload
        et, ea, tb = rep.payload
        raise RemoteError(et, ea, tb)

    def submit(self, *args, **kwargs) -> Future:
        """Submit fn(*args, **kwargs) asynchronously; returns a Future."""
        cid = uuid.uuid4().hex
        self._req.put(_Req(cid, self._rep._base, "apply", "", args, kwargs))
        return Future(self._mbox, cid)

    def map(self, iterable, *, timeout: Optional[float] = None):
        """Map fn over an iterable; returns a list of results (preserves order)."""
        cids: List[str] = []
        for x in iterable:
            cid = uuid.uuid4().hex
            self._req.put(_Req(cid, self._rep._base, "apply", "", (x,), {}))
            cids.append(cid)
        out: List[Any] = []
        for cid in cids:
            rep = self._mbox.wait(cid, timeout)
            if rep.ok:
                out.append(rep.payload)
            else:
                et, ea, tb = rep.payload
                raise RemoteError(et, ea, tb)
        return out

    def close(self, timeout: float = 2.0) -> None:
        """Gracefully stop all workers."""
        for _ in self._ps:
            cid = uuid.uuid4().hex
            self._req.put(_Req(cid, self._rep._base, "shutdown", "", (), {}))
        deadline = time.time() + timeout
        for p in self._ps:
            rem = max(0.0, deadline - time.time())
            p.join(rem)
            if p.is_alive():
                p.terminate()
                p.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close(timeout=1.0)
        except Exception:
            pass


def ActorDecorator(cls, **decorator_kwargs) -> Actor:
    """Decorator to create an Actor from a class definition.

    Args:
      cls: Class to run in the actor process (nested/local supported via cloudpickle).
      buffer_bytes: Size of each queue (request/mailbox) in bytes. Defaults to 10e6.
      **decorator_kwargs: Keyword args for the class constructor.

    Returns:
        A factory function that creates an Actor instance when called with positional
        and keyword arguments for the class constructor

    Example:
        @ActorDecorator(buffer_bytes=1e6)
        class Counter:
            def __init__(self, start=0):
                self.value = start
            def inc(self, n=1):
                self.value += n
                return self.value
            def get(self):
                return self.value
    """

    def WrappedActor(*args, **kwargs):
        kwargs = {**kwargs, **decorator_kwargs}
        return Actor(cls, *args, **kwargs)

    return WrappedActor


class PickledObject:
    """Decorator to create a pickled object from a class definition."""

    def __init__(self, cls: type):
        self._cls = cls
        self._pkl = cloudpickle.dumps(cls)

    def load(self):
        return cloudpickle.loads(self._pkl)
