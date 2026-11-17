"""Regression tests for the findings of the pre-0.9 review of dejaq.queues (numbered as in the review).

#14 (get_nowait() waits while another consumer reads an item with a callback) is documented behaviour.
"""

import gc
import os
import subprocess
import sys
import threading
import weakref

import numpy as np
import pytest

from dejaq import queues
from dejaq.queues import _IS_WIN, DejaQueue, ShutDown

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POSIX_ONLY = pytest.mark.skipif(_IS_WIN, reason="no resource tracker on Windows")


def _run_script(tmp_path, code, *args, timeout=60):
    """Run code as a script file (needed for the spawn start method), in a fresh interpreter."""
    script = tmp_path / "script.py"
    script.write_text(code)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          timeout=timeout, env=env)


# ---------- 1: a put in progress during shutdown() ----------

def test_1_put_in_progress_during_shutdown_is_not_committed():
    """Formerly the put returned True after consumers had already stopped: its item was never read."""
    q = DejaQueue(1_000_000)
    copying, proceed = threading.Event(), threading.Event()
    write = q._write_bytes

    def slow_write(*args, **kwargs):
        copying.set()
        proceed.wait(5)
        return write(*args, **kwargs)

    q._write_bytes = slow_write
    result = {}

    def produce():
        try:
            result["put"] = q.put(np.ones(10))
        except ShutDown:
            result["put"] = "ShutDown"

    t = threading.Thread(target=produce)
    t.start()
    assert copying.wait(5)
    q.shutdown()
    with pytest.raises(ShutDown):
        q.get(timeout=1)  # the consumer stops: the queue is shut down and empty
    proceed.set()
    t.join(5)
    assert result["put"] == "ShutDown" and q.qsize() == 0


# ---------- 2: fork after unlink() ----------

_FORK_AFTER_UNLINK = """
import os, sys, threading
from dejaq.queues import DejaQueue
q = DejaQueue(4096)
q.unlink()
q._put_lock.acquire()
r, w = os.pipe()
pid = os.fork()
if pid == 0:
    out = []
    def try_lock():
        try:
            q._put_lock.acquire()
            out.append(b"S")   # got the lock the parent holds: shared
        except RuntimeError:
            out.append(b"E")   # refused (lock file could not be reopened)
    threading.Thread(target=try_lock, daemon=True).start()
    threading.Event().wait(0.5)
    os.write(w, out[0] if out else b"B")   # B: blocked, as it must be
    os._exit(0)
os.waitpid(pid, 0)
print(os.read(r, 1).decode())
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_2_forked_child_never_shares_the_lock_of_an_unlinked_queue(tmp_path):
    r = _run_script(tmp_path, _FORK_AFTER_UNLINK)
    assert r.returncode == 0, r.stderr
    result = r.stdout.strip()
    assert result == "E", "the child must refuse, not share the lock its parent holds"


# ---------- 3: purge() after shutdown() ----------

def test_3_purge_after_shutdown_keeps_the_wake_up_for_waiting_consumers():
    """Formerly purge() removed shutdown()'s wake-up, so consumers already waiting were never told."""
    q = DejaQueue(10_000)
    q.put(1)
    q.shutdown()
    q.purge()
    assert q._items.acquire(timeout=0)  # the wake-up count is still there
    q._items.release(1)
    out = []
    t = threading.Thread(target=lambda: out.append(_get_or_shutdown(q)))
    t.start()
    t.join(5)
    assert out == ["ShutDown"]


def _get_or_shutdown(q):
    try:
        return q.get(timeout=3)
    except ShutDown:
        return "ShutDown"


# ---------- 4: legacy positional get(timeout, peek_only) ----------

def test_4_legacy_positional_peek_keeps_its_meaning():
    """Formerly get(None, True) waited 1 s and then removed the item (instead of peeking)."""
    q = DejaQueue(10_000)
    q.put("x")
    with pytest.warns(DeprecationWarning, match="peek_only"):
        assert q.get(None, True) == "x"  # peek
    with pytest.warns(DeprecationWarning, match="peek_only"):
        assert q.get(1.0, True) == "x"  # peek with a timeout
    assert q.qsize() == 1
    with pytest.warns(DeprecationWarning, match="peek_only"):
        assert q.get(None, False) == "x"  # wait indefinitely, remove
    assert q.qsize() == 0


# ---------- 5: concurrent garbage-collector pauses ----------

def test_5_overlapping_gc_pauses_from_two_threads():
    """Formerly the first pause to end re-enabled the collector inside the other one, and the last one
    to end could leave it disabled for good."""
    gc.enable()
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def a():
        with queues._gc_paused():
            a_in.set()
            b_in.wait(5)
        a_out.set()

    def b():
        a_in.wait(5)
        with queues._gc_paused():
            b_in.set()
            a_out.wait(5)
            seen["paused_after_a_ended"] = not gc.isenabled()

    ts = [threading.Thread(target=a), threading.Thread(target=b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    try:
        assert seen["paused_after_a_ended"]
        assert gc.isenabled()
    finally:
        gc.enable()


# ---------- 6: tracker calls from several threads ----------

@_POSIX_ONLY
def test_6_tracker_calls_from_threads_stay_in_order(monkeypatch):
    """Formerly a thread's "unregister" could overtake its own "register" (the tracker then keeps a stale
    record and removes the name at exit, even after reuse)."""
    from multiprocessing import resource_tracker

    sent, lock = [], threading.Lock()

    def recorder(action):
        def record(name, rtype):
            with lock:
                sent.append((action, name))
            threading.Event().wait(0.0005)  # let other threads run while "sending"
        return record

    monkeypatch.setattr(resource_tracker, "register", recorder("register"))
    monkeypatch.setattr(resource_tracker, "unregister", recorder("unregister"))
    errors = []

    def work(k):
        try:
            for i in range(50):
                queues._tracker("register", f"/dq_test_{k}_{i}", "semaphore")
                queues._tracker("unregister", f"/dq_test_{k}_{i}", "semaphore")
        except Exception as e:  # e.g. IndexError from a racing flush
            errors.append(e)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    queues._flush_tracker_backlog()
    assert errors == []
    order = {}
    for action, name in sent:
        if name.startswith("/dq_test_"):  # (other objects' cleanup may call the tracker meanwhile)
            order.setdefault(name, []).append(action)
    assert len(order) == 300 and all(v == ["register", "unregister"] for v in order.values())


@_POSIX_ONLY
def test_6_unregister_waits_for_a_register_still_being_sent(monkeypatch):
    """Deterministic version: while one thread is still sending "register X", another thread's
    "unregister X" must not be sent first (formerly it was)."""
    from multiprocessing import resource_tracker

    sent = []
    sending, proceed = threading.Event(), threading.Event()

    def register(name, rtype):
        if name == "/dq_test_order":
            sending.set()
            proceed.wait(5)  # e.g. a slow pipe write
        sent.append(("register", name))

    monkeypatch.setattr(resource_tracker, "register", register)
    monkeypatch.setattr(resource_tracker, "unregister", lambda name, rtype: sent.append(("unregister", name)))
    t1 = threading.Thread(target=queues._tracker, args=("register", "/dq_test_order", "semaphore"))
    t1.start()
    assert sending.wait(5)
    t2 = threading.Thread(target=queues._tracker, args=("unregister", "/dq_test_order", "semaphore"))
    t2.start()
    t2.join(0.3)  # formerly: sent its "unregister" now, before the "register" was complete
    proceed.set()
    t1.join(5)
    t2.join(5)
    queues._flush_tracker_backlog()
    assert [a for a, n in sent if n == "/dq_test_order"] == ["register", "unregister"]


# ---------- 7: worker processes send their queued tracker calls ----------

_WORKER_FINALIZES_IN_GC = """
import gc, sys, multiprocessing as mp
from dejaq.queues import NamedSemaphore

def child():
    s = NamedSemaphore()
    s.cycle = s  # freed only by the garbage collector: its "unregister" is queued
    del s
    gc.collect()

if __name__ == "__main__":
    p = mp.get_context(sys.argv[1]).Process(target=child)
    p.start()
    p.join()
    print("ok")
"""


@_POSIX_ONLY
@pytest.mark.parametrize("start_method", __import__("multiprocessing").get_all_start_methods())
def test_7_worker_process_sends_queued_tracker_calls(tmp_path, start_method):
    """Formerly a multiprocessing worker exited (os._exit) with its queued "unregister" unsent: at exit the
    tracker reported a leaked semaphore."""
    r = _run_script(tmp_path, _WORKER_FINALIZES_IN_GC, start_method)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    assert r.stderr.strip() == "", r.stderr


# ---------- 8: forked children do not send the parent's queued calls ----------

_FORK_WITH_QUEUED_CALLS = """
import gc, os
from dejaq.queues import NamedSemaphore
s = NamedSemaphore()
s.cycle = s
del s
gc.collect()  # the parent's "unregister" is queued
pid = os.fork()
if pid == 0:
    t = NamedSemaphore()  # sends this process's queued calls
    del t
    os._exit(0)
os.waitpid(pid, 0)
t = NamedSemaphore()  # sends the parent's queued call
del t
print("ok")
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_8_forked_child_does_not_resend_the_parents_queued_calls(tmp_path):
    """Formerly both parent and child sent the same "unregister": the tracker printed a KeyError."""
    r = _run_script(tmp_path, _FORK_WITH_QUEUED_CALLS)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    assert r.stderr.strip() == "", r.stderr


# ---------- 9: re-creating a fixed-name queue ----------

_RECREATE_THEN_CRASH = """
import gc, os, signal, sys
from dejaq.queues import DejaQueue
name = sys.argv[1]
q = DejaQueue(4096, name=name)
q.cycle = q  # finalized by the garbage collector: its "unregister" calls are queued
del q
gc.collect()
q = DejaQueue(4096, name=name)  # the same name again
print("created", flush=True)
os.kill(os.getpid(), signal.SIGKILL)
"""


@_POSIX_ONLY
def test_9_recreated_queue_is_still_cleaned_up_after_a_crash(tmp_path):
    """Formerly the old queue's queued "unregister" cancelled the new queue's registration, so its shared
    memory was left behind after a crash."""
    from test_teardown import _IS_MAC, _wait_for_no_leftovers, queue_leftovers

    name = "dq_test_" + os.urandom(5).hex()
    r = _run_script(tmp_path, _RECREATE_THEN_CRASH, name)
    assert "created" in r.stdout, r.stderr
    _, _, lock_names, _ = queues.NamedByteRing._names(name)
    allowed = set(lock_names) if _IS_MAC else set()  # macOS keeps a crashed creator's lock files
    try:
        assert _wait_for_no_leftovers(lambda: queue_leftovers(name), allowed=allowed) == []
    finally:
        for nm in allowed:
            try:
                os.unlink(queues._RobustLock._path_for(nm))
            except OSError:
                pass


# ---------- 10: failed creation with auto_unlink=False ----------

_FAILED_PERSISTENT_CREATE = """
from dejaq.queues import DejaQueue
try:
    DejaQueue(0, auto_unlink=False)  # the state segment is created, the buffer (size 0) fails
except ValueError:
    print("ok")
"""


@_POSIX_ONLY
def test_10_failed_creation_leaves_no_stale_registration(tmp_path):
    """Formerly the cleanup removed the state segment without unregistering it: warnings at exit."""
    r = _run_script(tmp_path, _FAILED_PERSISTENT_CREATE)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    assert r.stderr.strip() == "", r.stderr


# ---------- 11: one thread's garbage collection does not hold back other threads ----------

@_POSIX_ONLY
def test_11_other_threads_send_tracker_calls_while_one_collects(monkeypatch):
    """Formerly any thread's collection made every thread only queue its calls (crash cleanup then misses
    resources registered meanwhile)."""
    from multiprocessing import resource_tracker

    sent = []
    monkeypatch.setattr(resource_tracker, "register", lambda name, rtype: sent.append(name))
    in_finalizer, done = threading.Event(), threading.Event()

    class Cycle:
        pass

    def collector():
        c = Cycle()
        c.self = c
        weakref.finalize(c, lambda: (in_finalizer.set(), done.wait(5)))
        del c
        gc.collect()  # runs the finalizer, which waits while the main thread registers

    t = threading.Thread(target=collector)
    t.start()
    try:
        assert in_finalizer.wait(5)
        queues._tracker("register", "/dq_test_thread", "semaphore")
        assert "/dq_test_thread" in sent  # sent at once, not queued behind the other thread's collection
    finally:
        done.set()
        t.join(5)


# ---------- 12: unlink() continues after an error ----------

@_POSIX_ONLY
def test_12_unlink_removes_everything_it_can_and_reports_the_error(monkeypatch):
    import _posixshmem

    from test_teardown import queue_leftovers

    q = DejaQueue(4096)
    st_name, _, _, _ = queues.NamedByteRing._names(q._base)
    real = _posixshmem.shm_unlink

    def refuse_state_segment(name):
        if name == "/" + st_name:
            raise PermissionError(13, "Permission denied", name)
        return real(name)

    monkeypatch.setattr(_posixshmem, "shm_unlink", refuse_state_segment)
    with pytest.raises(PermissionError):
        q.unlink()
    monkeypatch.undo()
    try:
        assert queue_leftovers(q._base) == [st_name]  # the buffer, semaphores and lock files are gone
    finally:
        real("/" + st_name)


# ---------- 13: stop signals on a shut-down queue ----------

def test_13_stop_signal_after_shutdown_is_a_no_op():
    q = DejaQueue(10_000)
    q.put(1)
    q.shutdown()
    assert list(q) == [1]
    q._signal_stop(3)  # e.g. dejaq.parallel ending its iteration: formerly raised ShutDown


# ---------- 15: put() after shutdown ----------

class _CountsPickling:
    calls = 0

    def __reduce__(self):
        type(self).calls += 1
        return (_CountsPickling, ())


def test_15_put_after_shutdown_raises_before_pickling():
    q = DejaQueue(10_000)
    q.shutdown()
    with pytest.raises(ShutDown):
        q.put(_CountsPickling())
    assert _CountsPickling.calls == 0
    with pytest.raises(ShutDown):  # formerly ValueError (too large for the queue)
        q.put(b"x" * 20_000)
