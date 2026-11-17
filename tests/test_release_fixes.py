"""Regression tests for the findings of the final 1.0 reviews."""

import os
import sys
import threading
import time
import warnings

import numpy as np
import pytest

from dejaq import Actor, DejaQueue, Parallel, RemoteFunc
from dejaq.parallel import lazymap, zip as dzip

from _testutil import run_script


class Slow:
    def __init__(self, path=None):
        self.path = path
        self.ticks = 0

    def wait(self, s):
        time.sleep(s)
        return "done"

    def nop(self):
        return 1

    def total(self, a):
        return float(a.sum())

    def keep(self, a):
        self.kept = a[:2]
        return 0

    def tick(self):
        self.ticks += 1
        raise OSError("camera timeout")

    def n_ticks(self):
        return self.ticks

    def close(self):
        if self.path:
            time.sleep(1.5)
            open(self.path, "w").write("closed")


def test_future_result_can_be_retried_after_a_timeout():
    with Actor(Slow) as a:
        fut = a.wait_async(0.5)
        with pytest.raises(TimeoutError):
            fut.result(timeout=0.05)
        assert fut.result(timeout=5) == "done"


def test_discarded_futures_leave_no_replies_behind():
    with Actor(Slow) as a:
        for _ in range(200):
            a.nop_async()
        assert a.nop() == 1
        time.sleep(0.2)
        a.nop()
        assert len(a._mbox._got) == 0


def test_actor_close_waits_for_a_slow_close(tmp_path):
    path = tmp_path / "closed.txt"
    with Actor(Slow, str(path)):
        pass
    assert path.read_text() == "closed"


def test_an_idle_function_that_raises_does_not_kill_the_actor():
    class Idle(Slow):
        _idle_function = "tick"
        _idle_timeout = 0.05

    with Actor(Idle) as a:
        time.sleep(0.5)
        assert a.n_ticks() > 2


def test_retention_warning_only_when_a_reference_is_kept():
    with Actor(Slow) as a:
        x = np.ones(1000)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a.total(x, deepcopy=False)
            a.nop()
        assert not [m for m in w if "kept a reference" in str(m.message)]


def test_replies_of_other_threads_are_not_delayed():
    with RemoteFunc(time.sleep, workers=2) as rf:
        t = threading.Thread(target=rf, args=(2.0,))
        t.start()
        time.sleep(0.2)
        worst = 0.0
        for _ in range(30):
            t0 = time.perf_counter()
            rf(0)
            worst = max(worst, time.perf_counter() - t0)
        t.join()
    assert worst < 0.25


@pytest.mark.skipif(sys.platform == "win32", reason="forkserver is POSIX-only")
def test_forkserver_start_method():
    with Actor(Slow, start_method="forkserver") as a:
        assert a.nop() == 1


def test_wrapped_zero_copy_items_are_writable():
    q = DejaQueue(10 * 8000 + 500)
    for i in range(25):
        q.put(np.zeros(1000))
        q.get(callback=lambda a: a.__iadd__(1))  # in place, also when the item wraps the ring's end


def test_zip_refuses_the_same_stage_twice_and_sizes_its_buffer():
    s = Parallel(n_workers=1)(lambda x: x)(range(3))
    with pytest.raises(ValueError):
        dzip(s, s)
    t = Parallel(n_workers=1, buffer_bytes=20e6)(lambda x: x)(range(3))
    z = dzip(s, t)
    assert z._buffer_bytes == 30e6
    assert list(z) == [(0, 0), (1, 1), (2, 2)]


def test_a_stage_does_not_pickle_its_input():
    import cloudpickle
    s = lazymap(lambda x: x, list(range(100_000)))
    assert len(cloudpickle.dumps(s)) < 10_000
    assert list(s)[-1] == 99_999


def test_dropping_a_tee_branch_does_not_stall_the_others():
    a, b = Parallel(n_workers=1, buffer_bytes=1e5)(lambda x: np.zeros(1000) + x)(range(200)).tee(2)
    del b
    assert sum(1 for _ in a) == 200


WRITER = """
import sys, time
from dejaq import Parallel

class Writer:
    def __init__(self, path):
        self.path = path
    def __call__(self, x):
        return x
    def __del__(self):
        time.sleep(0.5)
        open(self.path, "w").write("finalized")

if __name__ == "__main__":
    with Parallel(n_workers=1)(Writer)(sys.argv[1])(range(5)) as s:
        assert s.run() == list(range(5))
"""


def test_finished_workers_may_finalize_after_close(tmp_path):
    path = tmp_path / "out.txt"
    r = run_script(tmp_path, WRITER, str(path))
    assert r.returncode == 0, r.stderr
    assert path.read_text() == "finalized"


EXIT = """
import sys, time
from dejaq import Actor

class Res:
    def __init__(self, path):
        self.path = path
    def hello(self):
        return 1
    def close(self):
        open(self.path, "w").write("closed")

if __name__ == "__main__":
    actors = [Actor(Res, sys.argv[1] + str(i)) for i in range(3)]
    assert all(a.hello() == 1 for a in actors)
"""


def test_unclosed_actors_are_closed_at_exit(tmp_path):
    t0 = time.monotonic()
    r = run_script(tmp_path, EXIT, str(tmp_path / "res"))
    assert r.returncode == 0, r.stderr
    assert [(tmp_path / f"res{i}").read_text() for i in range(3)] == ["closed"] * 3
    assert time.monotonic() - t0 < 8


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs fork and the semaphore's value")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_two_waiters_do_not_both_take_a_dead_holders_state_lock_over():
    for _ in range(3):
        q = DejaQueue(100_000)
        pid = os.fork()
        if pid == 0:  # dies holding the state lock
            q._state_lock.__enter__()
            os._exit(0)
        os.waitpid(pid, 0)
        go = time.time() + 0.2
        kids = []
        for _ in range(2):
            k = os.fork()
            if k == 0:
                time.sleep(max(0.0, go - time.time()))
                q.qsize()
                os._exit(0)
            kids.append(k)
        for k in kids:
            os.waitpid(k, 0)
        assert q._state_lock._sem.value == 1
        q.close()
