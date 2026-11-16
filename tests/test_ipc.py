# test_named_ipc.py
import os, time, pickle, multiprocessing as mp, numpy as np, pytest

# adjust this import to your module path
from dejaq.queues import _IS_WIN, NamedSemaphore, NamedLock, NamedByteRing, DejaQueue

mp.set_start_method("spawn", force=True)

# ---------- spawn-safe worker targets ----------

def _child_sem_acquire(name):
    try:
        from dejaq.queues import NamedSemaphore
        s = NamedSemaphore(name, create=False)
        ok = s.acquire(timeout=1.0)
        os._exit(0 if ok else 1)
    except Exception as e:
        import traceback; traceback.print_exc()
        os._exit(2)


def _child_nbr_get_bytes(base):
    try:
        from dejaq.queues import NamedByteRing
        q = NamedByteRing(name=base, create=False)
        ok, b = q.get_bytes(timeout=2.0)
        if not ok or b != b"hello":
            os._exit(1)
        os._exit(0)
    except Exception as e:
        import traceback; traceback.print_exc()
        os._exit(2)

def _child_dejaq_prefill_and_get(base, nbytes):
    try:
        from dejaq.queues import DejaQueue
        q = DejaQueue(name=base, create=False)
        # consume the prefill
        _ = q.get(timeout=2.0)
        # then the measured payload
        arr = q.get(timeout=2.0)
        ok = isinstance(arr, (bytes, bytearray, memoryview)) and len(arr) == nbytes
        os._exit(0 if ok else 1)
    except Exception as e:
        import traceback; traceback.print_exc()
        os._exit(2)

# ---------- tests ----------

def test_named_semaphore_roundtrip_spawn():
    ctx = mp.get_context("spawn")
    s = NamedSemaphore(create=True, initial=0)
    p = ctx.Process(target=_child_sem_acquire, args=(s.name,))
    p.start()
    time.sleep(0.1)
    s.release(1)
    p.join(2.0)
    assert p.exitcode == 0

def test_named_semaphore_semantics():
    s = NamedSemaphore(create=True, initial=1, maxcount=2)
    assert s.acquire(timeout=0) is True
    t0 = time.monotonic()
    assert s.acquire(timeout=0.2) is False
    assert time.monotonic() - t0 >= 0.15
    s.release(2)  # back to maxcount
    if _IS_WIN:  # maxcount is only enforced by Windows semaphores
        with pytest.raises(RuntimeError):
            s.release(1)  # over-release
    s2 = pickle.loads(pickle.dumps(s))  # attach by name
    assert s2.name == s.name
    assert s2.acquire(timeout=0) and s.acquire(timeout=0)
    assert not s2.acquire(timeout=0)
    s2.close()
    s.close()


def test_named_lock_over_release():
    lk = NamedLock(create=True)
    with lk:
        pass
    with pytest.raises(RuntimeError):
        lk.release()
    lk.close()


_EXIT_CHILD = """
from dejaq.queues import DejaQueue, NamedByteRing
q = DejaQueue(1_000_000)
q.put({"a": 1})
assert q.get() == {"a": 1}
q2 = NamedByteRing(1_000_000)
print("ok")
"""  # queues deliberately left alive until interpreter shutdown


def test_queue_alive_at_exit_closes_cleanly(tmp_path):
    """No 'BufferError: cannot close exported pointers exist' when a queue lives until shutdown (seen on 3.14t)."""
    import subprocess, sys

    # run as a script file: with `python -c` the shutdown order differs and the bug does not show
    script = tmp_path / "child.py"
    script.write_text(_EXIT_CHILD)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [repo_root, os.environ.get("PYTHONPATH")])))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    assert "BufferError" not in r.stderr and "Exception ignored" not in r.stderr, r.stderr


def test_named_semaphore_open_missing():
    if _IS_WIN:
        expected = FileNotFoundError
    else:
        import posix_ipc
        expected = posix_ipc.ExistentialError
    with pytest.raises(expected):
        NamedSemaphore("dejaq_does_not_exist_" + os.urandom(4).hex(), create=False)


@pytest.mark.skipif(not _IS_WIN, reason="Windows handle semantics")
def test_named_semaphore_use_after_close():
    a = NamedSemaphore(initial=0)
    a.close()
    a.close()  # idempotent
    others = [NamedSemaphore(initial=0) for _ in range(8)]  # likely to reuse a's old handle value
    with pytest.raises(ValueError):
        a.release()
    with pytest.raises(ValueError):
        a.acquire(timeout=0)
    assert not any(o.acquire(timeout=0) for o in others)  # no other semaphore was touched
    for o in others:
        o.close()


@pytest.mark.skipif(not _IS_WIN, reason="Windows enforces 32-bit LONG counts")
def test_named_semaphore_count_range():
    s = NamedSemaphore(initial=0, maxcount=10)
    for bad in (0, -1, 2**31, 2**32 + 1):
        with pytest.raises(ValueError):
            s.release(bad)
    assert not s.acquire(timeout=0)  # nothing was released by the rejected calls
    with pytest.raises(ValueError):
        NamedSemaphore(maxcount=2**31)
    with pytest.raises(ValueError):
        NamedSemaphore(initial=5, maxcount=2)
    s.close()


def test_picklable_dejaqueue_prefill_then_get():
    """Reproduces the 'prefill then bench' flow."""
    ctx = mp.get_context("spawn")
    q = DejaQueue(buffer_bytes=2_000_000, create=True)
    base = q._base
    # prefill with a small object
    q.put(1)
    # spawn consumer that drains prefill and then expects the big item
    p = ctx.Process(target=_child_dejaq_prefill_and_get, args=(base, 1024))
    p.start()
    q.put(np.random.bytes(1024))  # second item
    p.join(5.0)
    assert p.exitcode == 0
