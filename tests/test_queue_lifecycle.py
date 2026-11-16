"""Queue lifecycle: processes that die while using a queue, and when a queue's resources are removed."""

import os
import subprocess
import sys
import threading
import time
import multiprocessing as mp

import numpy as np
import pytest

from dejaq import queues
from dejaq.queues import _IS_WIN, DejaQueue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ctx = mp.get_context("spawn")


def _run_python(code, timeout=60):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout, env=env)


def _call_with_timeout(fn, timeout):
    """Run fn in a thread; fail the test (instead of hanging it) if it does not finish in time."""
    res = {}

    def run():
        try:
            res["value"] = fn()
        except BaseException as e:
            res["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), f"blocked for more than {timeout} s"
    if "error" in res:
        raise res["error"]
    return res["value"]


def _assert_idle_and_consistent(q):
    with pytest.raises(TimeoutError):
        q.get(timeout=0.05)
    assert q.nitems == 0 and q.is_empty
    q.put("probe")
    assert q.get(timeout=1) == "probe"


# ---------- processes dying while holding a lock ----------

def _get_and_hang(q, ready):
    def hang(obj):  # runs while the get lock is held
        ready.set()
        time.sleep(600)

    q.get(callback=hang)


def _put_and_hang(q, ready):
    def hang(*args, **kwargs):  # runs while the put lock is held, after part of the item may be written
        ready.set()
        time.sleep(600)

    q._write_bytes = hang
    q.put(np.arange(1000))


def _start_and_kill(target, q):
    ready = ctx.Event()
    p = ctx.Process(target=target, args=(q, ready))
    p.start()
    assert ready.wait(60), "child did not reach the lock"
    p.kill()  # SIGKILL / TerminateProcess: no cleanup code runs in the child
    p.join(30)


def test_consumer_killed_while_holding_get_lock():
    q = DejaQueue(100_000)
    for item in ("first", "second", "third"):
        q.put(item)
    _start_and_kill(_get_and_hang, q)
    # the item the dead consumer was reading is delivered again, nothing is lost or duplicated
    assert _call_with_timeout(lambda: [q.get(timeout=5) for _ in range(3)], 30) == ["first", "second", "third"]
    _assert_idle_and_consistent(q)


def test_producer_killed_while_holding_put_lock():
    q = DejaQueue(100_000)
    q.put("before")
    _start_and_kill(_put_and_hang, q)
    _call_with_timeout(lambda: q.put("after", timeout=5), 30)
    assert q.get(timeout=1) == "before"
    assert q.get(timeout=1) == "after"  # the dead producer's partial item never became visible
    _assert_idle_and_consistent(q)


def test_killed_consumer_while_another_consumer_waits():
    """A consumer already waiting for the lock takes over once the holder is killed."""
    q = DejaQueue(100_000)
    q.put("first")
    q.put("second")
    ready = ctx.Event()
    p = ctx.Process(target=_get_and_hang, args=(q, ready))
    p.start()
    assert ready.wait(60)
    got = []
    t = threading.Thread(target=lambda: got.append(q.get(timeout=30)), daemon=True)
    t.start()
    time.sleep(0.5)
    assert not got  # waiting for the lock
    p.kill()
    p.join(30)
    t.join(30)
    assert got == ["first"]
    assert q.get(timeout=1) == "second"
    _assert_idle_and_consistent(q)


_FORK_LOCK_CHILD = """
import os, threading, time
from dejaq.queues import DejaQueue
q = DejaQueue(4096)
q._get_lock.acquire()  # held by the parent across the fork
pid = os.fork()
if pid == 0:
    got = threading.Event()
    threading.Thread(target=lambda: (q._get_lock.acquire(), got.set()), daemon=True).start()
    blocked = not got.wait(0.5)  # must not get the parent's lock
    os.write(w, b"1" if blocked and got.wait(10) else b"0")
    os._exit(0)
""".replace("pid = os.fork()", "r, w = os.pipe()\npid = os.fork()")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_does_not_share_the_parents_lock():
    code = _FORK_LOCK_CHILD + """
time.sleep(1.0)
q._get_lock.release()  # now the child gets it
os.waitpid(pid, 0)
print("ok" if os.read(r, 1) == b"1" else "shared")
"""
    r = _run_python(code)
    assert r.returncode == 0 and r.stdout.strip() == "ok", (r.stdout, r.stderr)


# ---------- explicit lifecycle ----------

def test_close_then_unlink_and_repeated_calls():
    q = DejaQueue(4096)
    name = q._base
    q.put(1)
    q.close()
    q.close()
    q.unlink()  # formerly AttributeError after close()
    q.unlink()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        DejaQueue(name=name, create=False)


def test_unlink_does_not_disturb_attached_users():
    q = DejaQueue(4096)
    q2 = DejaQueue(name=q._base, create=False)
    q.unlink()
    q.put("still works")
    assert q2.get(timeout=1) == "still works"


def test_context_manager():
    with DejaQueue(4096) as q:
        name = q._base
        with DejaQueue(name=name, create=False):  # an attacher's exit does not remove the queue
            q.put(1)
        assert q.get(timeout=1) == 1
        DejaQueue(name=name, create=False)
    with pytest.raises(FileNotFoundError):
        DejaQueue(name=name, create=False)


def test_queue_is_removed_when_creator_object_is_gone():
    import gc

    q = DejaQueue(4096)
    name = q._base
    del q
    gc.collect()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        DejaQueue(name=name, create=False)


_PERSISTENT_CHILD = """
from dejaq.queues import DejaQueue
q = DejaQueue(4096, name=NAME, auto_unlink=False)
q.put("kept")
print("ok")
"""


@pytest.mark.skipif(_IS_WIN, reason="Windows removes kernel objects with their last handle")
def test_auto_unlink_false_outlives_the_creating_process():
    name = "dq_test_" + os.urandom(5).hex()
    r = _run_python(_PERSISTENT_CHILD.replace("NAME", repr(name)))
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    assert "resource_tracker" not in r.stderr, r.stderr
    q = DejaQueue(name=name, create=False)
    assert q.get(timeout=1) == "kept"
    q.unlink()  # any process can remove a persistent queue
    with pytest.raises(FileNotFoundError):
        DejaQueue(name=name, create=False)


_CRASHING_CHILD = """
import os, signal
from dejaq.queues import DejaQueue
q = DejaQueue(4096, name=NAME)
q.put("x")
print("created", flush=True)
os.kill(os.getpid(), signal.SIGKILL)
"""


@pytest.mark.skipif(_IS_WIN, reason="Windows removes kernel objects with their last handle")
def test_crashed_creator_leaves_nothing_behind():
    """The resource tracker removes shared memory, semaphores and (on Linux) lock files of a crashed creator."""
    import posix_ipc

    name = "dq_test_" + os.urandom(5).hex()
    r = _run_python(_CRASHING_CHILD.replace("NAME", repr(name)))
    assert "created" in r.stdout, r.stderr
    _, _, lock_names, sem_names = queues.NamedByteRing._names(name)

    def leftovers():
        left = []
        try:
            DejaQueue(name=name, create=False)
            left.append("shared memory")
        except FileNotFoundError:
            pass
        for nm in sem_names:
            try:
                posix_ipc.Semaphore(queues._posix_name(nm)).close()
                left.append(nm)
            except posix_ipc.ExistentialError:
                pass
        if queues._LOCK_FILES_TRACKABLE:
            left += [nm for nm in lock_names if os.path.exists(queues._RobustLock._path_for(nm))]
        return left

    deadline = time.monotonic() + 15
    while leftovers() and time.monotonic() < deadline:
        time.sleep(0.2)  # the tracker cleans up shortly after noticing the process is gone
    assert leftovers() == []


_CLEAN_EXIT_CHILD = """
import gc
from dejaq.queues import DejaQueue, NamedSemaphore
keep = [DejaQueue(4096), NamedSemaphore()]   # alive until interpreter exit
for _ in range(20):
    q = DejaQueue(4096); q.put(1); q.get(); s = NamedSemaphore()
del q, s
gc.collect()
print("ok")
"""


def test_no_resource_warnings_on_normal_exit():
    r = _run_python(_CLEAN_EXIT_CHILD)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    assert r.stderr.strip() == "", r.stderr  # e.g. resource_tracker "leaked semaphore" warnings
