"""dejaq.remote: dead and closed actors, threads, actors passed to other processes, full queues, zero copy,
exception types, stopping. Anything that used to hang runs with a time limit."""

import gc
import math
import os
import threading
import time

import numpy as np
import psutil
import pytest

from dejaq import Actor, ActorDecorator, Parallel, RemoteFunc
from dejaq.remote import RemoteError
from _testutil import all_gone, run_script


class Svc:
    def __init__(self, fail=False):
        if fail:
            raise ValueError("bad constructor")
        self.x = 0

    def inc(self, k=1):
        self.x += k
        return self.x

    def pid(self):
        return os.getpid()

    def die(self):
        os._exit(3)

    def slow(self, s):
        time.sleep(s)
        return s

    def raise_key(self):
        raise KeyError("missing")

    def raise_unpicklable(self):
        raise ValueError(threading.Lock())

    def return_lock(self):
        return threading.Lock()

    def echo(self, arr):
        return arr

    def every_other(self, arr):
        return arr[::2]

    def keep_view(self, arr):
        self.kept = arr[1:]
        return 0


def square(x):
    return x * x


def fails_at_3(x):
    if x == 3:
        raise ValueError("bad item 3")
    return x


def with_timeout(fn, timeout=30):
    """Run fn in a thread; fail the test (instead of hanging) if it does not finish in time."""
    out = []
    t = threading.Thread(target=lambda: out.append(_capture(fn)), daemon=True)
    t.start()
    t.join(timeout)
    assert out, f"hung for {timeout} s"
    kind, value = out[0]
    if kind == "error":
        raise value
    return value


def _capture(fn):
    try:
        return "ok", fn()
    except BaseException as e:
        return "error", e


# ---------- dead and closed actors ----------

def test_call_to_a_killed_actor_raises():
    a = Actor(Svc)
    psutil.Process(a.pid()).kill()
    with pytest.raises(RemoteError, match="exited"):
        with_timeout(a.inc)


def test_actor_that_dies_in_a_call_raises():
    a = Actor(Svc)
    with pytest.raises(RemoteError, match="exited"):
        with_timeout(a.die)


def test_call_after_close_raises():
    a = Actor(Svc)
    a.close()
    with pytest.raises(RemoteError, match="closed"):
        a.inc()


def test_constructor_error_is_raised_by_calls():
    with Actor(Svc, fail=True) as a:
        with pytest.raises(ValueError, match="bad constructor"):
            with_timeout(a.inc)


# ---------- exceptions ----------

def test_remote_exceptions_keep_their_type():
    with Actor(Svc) as a:
        with pytest.raises(KeyError, match="missing") as info:
            a.raise_key()
        assert "raise_key" in str(info.value.__cause__)  # the remote traceback
        with pytest.raises(RemoteError, match="ValueError"):
            a.raise_unpicklable()  # cannot be passed on: RemoteError with type and message
        with pytest.raises(TypeError, match="pickle"):
            a.return_lock()  # a result that cannot be passed on
        assert a.inc() == 1  # still serving


def test_remote_func_errors_keep_their_type_and_workers_survive():
    with RemoteFunc(fails_at_3, workers=2) as rf:
        with pytest.raises(ValueError, match="bad item 3"):
            rf(3)
        assert rf.map(range(6)[:3]) == [0, 1, 2] and rf(4) == 4


# ---------- threads, futures, timeouts ----------

def test_threads_share_an_actor():
    with Actor(Svc) as a:
        out = []

        def work():
            out.extend(a.inc() for _ in range(200))

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert sorted(out) == list(range(1, 801))


def test_future_result_can_be_read_twice():
    with Actor(Svc) as a:
        fut = a.inc_async(5)
        assert fut.result(10) == 5 and fut.result() == 5


def test_timeout_and_late_reply():
    with Actor(Svc) as a:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            a.slow(1.0, timeout=0.2)
        assert time.monotonic() - t0 < 1.0
        assert a.inc() == 1  # the late reply of the slow call is discarded, not mistaken for this one
        assert not a._mbox._got


def test_attributes_of_a_busy_actor_wait_instead_of_timing_out():
    with Actor(Svc) as a:
        busy = threading.Thread(target=a.slow, args=(2.5,))
        busy.start()
        time.sleep(0.2)
        a.x = 5  # used to fail after 2 s (and be applied later anyway)
        assert a.x == 5
        busy.join()


# ---------- full queues ----------

def bytes_2k(x):
    return bytes(2000)


def test_map_larger_than_the_queues():
    with RemoteFunc(bytes_2k, workers=2, buffer_bytes=200_000) as rf:  # the queues hold ~100 items
        assert len(with_timeout(lambda: rf.map(range(3000)), 120)) == 3000


def test_many_async_calls_larger_than_the_queues():
    with Actor(Svc, buffer_bytes=100_000) as a:
        futs = with_timeout(lambda: [a.inc_async() for _ in range(5000)], 120)
        assert [f.result(60) for f in futs] == list(range(1, 5001))


# ---------- actors in other processes ----------

def bump(i, actor):
    return actor.inc()


def test_actor_passed_to_worker_processes():
    with Actor(Svc) as a:
        results = with_timeout(lambda: Parallel(n_workers=2)(bump)(range(20), actor=a).run(progress=False), 120)
        assert sorted(results) == list(range(1, 21)) and a.inc() == 21


def test_temporary_actor_and_garbage_collection():
    assert Actor(Svc).inc() == 1  # (the method used to hold only a weak reference)
    a = Actor(Svc)
    pid = a.pid()
    del a
    gc.collect()
    assert all_gone([pid])


_NOT_CLOSED = """
from dejaq import Actor, RemoteFunc

def square(x):
    return x * x

class Svc:
    def get(self):
        return 1

if __name__ == "__main__":
    rf = RemoteFunc(square, workers=2)
    a = Actor(Svc)
    print(rf(3), a.get(), flush=True)
"""


def test_program_exits_without_close(tmp_path):
    r = run_script(tmp_path, _NOT_CLOSED)
    assert r.returncode == 0 and r.stdout.split() == ["9", "1"], r.stderr[-2000:]


# ---------- zero copy ----------

def test_zero_copy_results_stay_correct():
    with Actor(Svc, buffer_bytes=3_000_000) as a:
        for v in range(20):  # (the request queue is reused many times)
            arr = np.full(200_000, v)
            assert (a.echo(arr, deepcopy=False) == v).all()
            assert (a.every_other(arr, deepcopy=False) == v).all()


def test_zero_copy_retention_warning_covers_views(capfd):
    with Actor(Svc) as a:
        a.keep_view(np.ones(10), deepcopy=False)
    assert "kept a reference" in capfd.readouterr().err


# ---------- small things ----------

def test_decorator_with_options_and_builtin_functions():
    Counter = ActorDecorator(buffer_bytes=1e6)(Svc)
    with Counter() as c:
        assert c.inc(2) == 2
    with Actor(math) as m:
        assert m.log(8, 2) == 3.0  # (a builtin without a signature)


def test_remote_func_with_lambda_and_unpicklable_request():
    with RemoteFunc(lambda x: x + 1) as rf:
        assert rf(1) == 2
        with pytest.raises(TimeoutError):  # an argument the worker cannot unpickle: dropped, see the docstring
            rf(_Unpicklable(), timeout=1.0)
        assert rf(2) == 3


class _Unpicklable:
    def __reduce__(self):
        return (_fail, ())


def _fail():
    raise RuntimeError("cannot be unpickled")
