"""Teardown: no shared memory, semaphores, locks or lock files are left behind, on every platform.

Each test probes every named OS object of a queue (or semaphore) by name after a teardown path.
POSIX: shared-memory segments and semaphores persist until unlinked, lock files until deleted.
Windows: kernel objects exist while any process holds a handle, so a successful open by name after
teardown means a leaked handle.
"""

import gc
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pytest

from dejaq import queues
from dejaq.queues import _IS_WIN, DejaQueue, NamedSemaphore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IS_MAC = sys.platform == "darwin"

if _IS_WIN:
    _MISSING = (OSError,)
else:
    import posix_ipc

    _MISSING = (OSError, posix_ipc.ExistentialError)


# ---------- probes ----------

def _shm_exists(name: str) -> bool:
    try:
        shm = queues._attach_shm(name)  # does not register with the resource tracker
    except FileNotFoundError:
        return False
    shm.close()
    return True


def _sem_exists(name: str) -> bool:
    try:
        s = NamedSemaphore(name, create=False)
    except _MISSING:
        return False
    s.close()
    return True


def _lock_exists(name: str) -> bool:
    if not _IS_WIN:
        return os.path.exists(queues._RobustLock._path_for(name))
    try:
        lk = queues._RobustLock(name, create=False)
    except OSError:
        return False
    lk.close()
    return True


def queue_leftovers(base: str) -> list:
    """Names of all of the queue's OS objects that still exist."""
    st_name, buf_name, lock_names, sem_names = queues.NamedByteRing._names(base)
    left = [nm for nm in (st_name, buf_name) if _shm_exists(nm)]
    left += [nm for nm in sem_names if _sem_exists(nm)]
    left += [nm for nm in lock_names if _lock_exists(nm)]
    return left


def _wait_for_no_leftovers(probe, timeout=15.0, allowed=()):
    """Leftovers of a process that ended are removed shortly afterwards (by its resource tracker)."""
    deadline = time.monotonic() + timeout
    while True:
        left = [nm for nm in probe() if nm not in allowed]
        if not left or time.monotonic() > deadline:
            return left
        time.sleep(0.1)


def _run_python(code, timeout=60):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout, env=env)


def _used_queue(**kwargs):
    q = DejaQueue(4096, **kwargs)
    q.put(np.arange(10))
    q.get(timeout=1)
    return q


# ---------- in this process ----------

def test_probe_sees_a_live_queue():
    q = _used_queue()
    assert len(queue_leftovers(q._base)) == 7  # 2 segments, 3 semaphores, 2 locks


@pytest.mark.parametrize("how", ["gc", "close_then_gc", "unlink_then_gc", "close_unlink", "context_manager"])
def test_no_leftovers_after_teardown(how):
    q = _used_queue()
    base = q._base
    attached = pickle.loads(pickle.dumps(q))  # an attached object in this process: also released
    attached.put(1)
    assert q.get(timeout=1) == 1
    del attached
    if how == "gc":
        del q
    elif how == "close_then_gc":
        q.close()
        del q
    elif how == "unlink_then_gc":
        q.unlink()
        if not _IS_WIN:  # POSIX: names are gone at once; Windows: when the last handle closes
            assert queue_leftovers(base) == []
        del q
    elif how == "close_unlink":
        q.close()
        q.unlink()  # q stays referenced: everything must be gone anyway
    elif how == "context_manager":
        with DejaQueue(4096) as q:
            base = q._base
            q.put(1)
            q.get(timeout=1)
        # q is still referenced
    gc.collect()
    assert queue_leftovers(base) == []


def test_attached_object_keeps_working_after_creator_is_gone_then_is_released():
    q = _used_queue()
    base = q._base
    attached = DejaQueue(name=base, create=False)
    del q
    gc.collect()
    attached.put("still works")
    assert attached.get(timeout=1) == "still works"
    del attached
    gc.collect()
    assert queue_leftovers(base) == []


def test_standalone_semaphore_is_removed():
    s = NamedSemaphore()
    name = s.name
    s.release()
    assert _sem_exists(name)
    del s
    gc.collect()
    assert not _sem_exists(name)


def test_failed_attach_leaves_nothing_open():
    with pytest.raises(FileNotFoundError):
        DejaQueue(name="dq_test_" + os.urandom(5).hex(), create=False)
    q = _used_queue()
    base = q._base
    q.close()
    q.unlink()
    with pytest.raises(FileNotFoundError):
        DejaQueue(name=base, create=False)
    assert queue_leftovers(base) == []


# ---------- other processes ----------

_NORMAL_EXIT = """
import gc, numpy as np
from dejaq.queues import DejaQueue, NamedSemaphore
kept = DejaQueue(4096); kept.put(np.arange(3)); kept.get()   # alive until interpreter exit
dropped = DejaQueue(4096); dropped.put(1); dropped.get()
sem = NamedSemaphore()                                       # alive until interpreter exit
print(kept._base, dropped._base, sem.name)
del dropped; gc.collect()
"""


def test_no_leftovers_after_normal_exit():
    r = _run_python(_NORMAL_EXIT)
    assert r.returncode == 0, r.stderr
    assert r.stderr.strip() == "", r.stderr  # no resource-tracker warnings
    kept, dropped, sem = r.stdout.split()
    assert queue_leftovers(kept) == [] and queue_leftovers(dropped) == []
    assert not _sem_exists(sem)


_CRASH = """
import os, signal, sys
from dejaq.queues import DejaQueue, NamedSemaphore
q = DejaQueue(4096); q.put(1)
sem = NamedSemaphore()
print(q._base, sem.name, flush=True)
if sys.argv[1] == "exit":
    os._exit(1)                      # no cleanup code runs
elif hasattr(signal, "SIGKILL"):
    os.kill(os.getpid(), signal.SIGKILL)
else:
    os.kill(os.getpid(), signal.SIGTERM)  # Windows: TerminateProcess
"""


@pytest.mark.parametrize("how", ["exit", "kill"])
def test_no_leftovers_after_crash(how):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    r = subprocess.run([sys.executable, "-c", _CRASH, how], capture_output=True, text=True, timeout=60, env=env)
    base, sem = r.stdout.split()
    _, _, lock_names, _ = queues.NamedByteRing._names(base)
    # macOS: lock files of a crashed creator remain (empty files in /tmp; no resource tracker support)
    allowed = set(lock_names) if _IS_MAC else set()
    try:
        assert _wait_for_no_leftovers(lambda: queue_leftovers(base), allowed=allowed) == []
        assert _wait_for_no_leftovers(lambda: [sem] if _sem_exists(sem) else []) == []
    finally:
        for nm in allowed:
            try:
                os.unlink(queues._RobustLock._path_for(nm))
            except OSError:
                pass


_ATTACHER = """
import os, sys
from dejaq.queues import DejaQueue
q = DejaQueue(name=sys.argv[2], create=False)
q.put("from attacher")
if sys.argv[1] == "exit":
    os._exit(1)
"""


@pytest.mark.parametrize("how", ["normal", "exit"])
def test_attacher_exit_does_not_remove_the_queue(how):
    q = _used_queue()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    subprocess.run([sys.executable, "-c", _ATTACHER, how, q._base], timeout=60, env=env)
    time.sleep(0.5)  # give a resource tracker the chance to (wrongly) remove anything
    assert q.get(timeout=5) == "from attacher"
    assert len(queue_leftovers(q._base)) == 7
    DejaQueue(name=q._base, create=False).put(1)  # new processes can still attach
    assert q.get(timeout=1) == 1


_GC_INSIDE_TRACKER = """
import gc, warnings
from multiprocessing import resource_tracker
from dejaq.queues import DejaQueue, NamedSemaphore
warnings.simplefilter("always")

def garbage():  # objects that only the garbage collector can free
    q = DejaQueue(4096); q.put(1); q.get(); q.cycle = q
    s = NamedSemaphore(); s.cycle = s

tracker = resource_tracker._resource_tracker
tracker.ensure_running()
for _ in range(5):
    garbage()
    with tracker._lock:  # as if the collector ran while the tracker is busy in this thread
        gc.collect()
print("ok")
"""


@pytest.mark.skipif(_IS_WIN, reason="no resource tracker on Windows")
def test_garbage_collection_while_resource_tracker_is_busy():
    """Finalizers run by the garbage collector must not call into a busy resource tracker (Python >= 3.11.7:
    skipped with a warning, so the tracker's records go stale; earlier versions: deadlock)."""
    r = _run_python(_GC_INSIDE_TRACKER)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    assert r.stderr.strip() == "", r.stderr  # no reentrancy warnings, no "leaked" warnings at exit
