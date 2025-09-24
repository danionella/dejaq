# test_named_ipc.py
import os, time, multiprocessing as mp, numpy as np, pytest

# adjust this import to your module path
from dejaq.queues import _IS_WIN, NamedSemaphore, NamedByteRing, DejaQueue

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
