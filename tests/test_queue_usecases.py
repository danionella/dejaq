"""Fast, in-process tests of DejaQueue use cases that the multiprocess tests don't reach.

Tiny ring buffers make wraparound and a full queue happen within a few items, so the whole
file runs in a few seconds on every platform.
"""

import gc
import os
import sys
import threading
import time

import numpy as np
import pytest

from dejaq.queues import _IS_WIN, DejaQueue


def _roundtrip_equal(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    return a == b


# ---------- ring buffer mechanics ----------

def test_wraparound_with_numpy_buffers():
    """Items (header, pickle and out-of-band buffers) crossing the end of the ring stay intact."""
    q = DejaQueue(20_000)
    rng = np.random.default_rng(0)
    sent = 0
    for _ in range(60):
        batch = [{"i": sent + k, "a": rng.standard_normal(rng.integers(0, 600))} for k in range(3)]
        for item in batch:  # several items in flight, so headers and buffers land on the boundary too
            q.put(item, timeout=1)
        for item in batch:
            got = q.get(timeout=1)
            assert got["i"] == item["i"] and _roundtrip_equal(got["a"], item["a"])
        sent += 3
    assert q.nitems == 0 and q.is_empty


def test_full_queue_put_times_out_and_blocking_put_resumes():
    q = DejaQueue(10_000)
    q.put(b"x" * 6_000)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        q.put(b"y" * 6_000, timeout=0.2)
    assert time.monotonic() - t0 >= 0.15

    consumer = threading.Thread(target=lambda: (time.sleep(0.2), q.get(timeout=1)))
    consumer.start()
    q.put(b"z" * 6_000)  # blocks until the consumer frees space
    consumer.join()
    assert q.get(timeout=1) == b"z" * 6_000


def test_get_times_out_on_empty_queue():
    q = DejaQueue(10_000)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        q.get(timeout=0.2)
    assert time.monotonic() - t0 >= 0.15


def test_item_larger_than_buffer_is_rejected():
    q = DejaQueue(10_000)
    with pytest.raises(ValueError):
        q.put(np.zeros(2_000))  # 16 kB
    q.put("still works")
    assert q.get(timeout=1) == "still works"


def test_state_properties_and_purge():
    q = DejaQueue(10_000)
    assert q.is_empty and q.nitems == 0 and q.bytes_available == q.cap - 1
    for i in range(3):
        q.put(i)
    assert not q.is_empty and q.nitems == 3 and q.bytes_available < q.cap - 1
    q.purge()
    assert q.is_empty and q.nitems == 0
    with pytest.raises(TimeoutError):
        q.get(timeout=0.05)
    q.put("after purge")
    assert q.get(timeout=1) == "after purge"
    assert not q.closed
    q.closed = True
    assert q.closed


def test_peek_and_callback():
    q = DejaQueue(10_000)
    q.put(np.arange(5))
    assert np.array_equal(q.get(peek_only=True, timeout=1), np.arange(5))
    assert q.nitems == 1  # peek does not consume
    assert q.get(callback=lambda a: int(a.sum()), timeout=1) == 10
    assert q.nitems == 0


# ---------- payloads ----------

@pytest.mark.parametrize(
    "obj",
    [
        np.asfortranarray(np.arange(12.0).reshape(3, 4)),
        np.arange(100.0)[::3],  # non-contiguous
        np.arange(12).reshape(3, 4).T,
        np.zeros((0, 5)),  # empty buffer
        np.array([1, "a", None], dtype=object),
        np.zeros(3, dtype=[("x", "f4"), ("y", "i2")]),
        np.float32(1.5),
        {"nested": [np.ones(3), {"b": np.arange(4, dtype=np.uint8)}], "s": "text"},
    ],
    ids=["fortran", "noncontig", "transposed", "empty", "object", "structured", "scalar", "nested"],
)
def test_payload_roundtrip(obj):
    q = DejaQueue(100_000)
    q.put(obj)
    got = q.get(timeout=1)
    if isinstance(obj, dict):
        assert np.array_equal(got["nested"][0], obj["nested"][0])
        assert np.array_equal(got["nested"][1]["b"], obj["nested"][1]["b"]) and got["s"] == obj["s"]
    else:
        assert _roundtrip_equal(got, obj)


def _make_local():
    class Local:
        x = 3

    def double(v):
        return 2 * v

    return Local, double


def test_unpicklable_by_pickle_falls_back_to_cloudpickle():
    """Lambdas and local functions/classes, also next to out-of-band numpy buffers."""
    Local, double = _make_local()
    q = DejaQueue(100_000)
    q.put(lambda v: v + 1)
    assert q.get(timeout=1)(1) == 2
    q.put(double)
    assert q.get(timeout=1)(2) == 4
    q.put(Local())
    assert q.get(timeout=1).x == 3
    # pickle fails only after it has already emitted an out-of-band buffer
    q.put({"a": np.arange(10.0), "f": double, "b": np.ones(3)})
    got = q.get(timeout=1)
    assert np.array_equal(got["a"], np.arange(10.0)) and got["f"](3) == 6 and np.array_equal(got["b"], np.ones(3))


class _Unloadable:
    """Pickles fine, but raises when unpickled (like a class the consumer cannot import)."""

    def __reduce__(self):
        return (_raise_on_load, ())


def _raise_on_load():
    raise RuntimeError("cannot reconstruct")


def test_item_that_fails_to_unpickle_does_not_poison_queue():
    q = DejaQueue(10_000)
    for item in ("a", _Unloadable(), "c"):
        q.put(item)
    assert q.get(timeout=1) == "a"
    with pytest.raises(RuntimeError, match="cannot reconstruct"):
        q.get(timeout=1)
    assert q.get(timeout=1) == "c"  # the bad item was dropped, the next one is delivered
    assert q.nitems == 0 and q.is_empty
    q.put("d")
    assert q.get(timeout=1) == "d"


def test_failing_callback_does_not_poison_queue():
    q = DejaQueue(10_000)
    q.put(1)
    q.put(2)

    def boom(_):
        raise KeyError("callback failed")

    with pytest.raises(KeyError):
        q.get(callback=boom, timeout=1)
    assert q.get(timeout=1) == 2


def test_failing_peek_keeps_item():
    q = DejaQueue(10_000)
    q.put(_Unloadable())
    q.put("next")
    with pytest.raises(RuntimeError):
        q.get(peek_only=True, timeout=1)
    assert q.nitems == 2  # peek never consumes, even when it fails
    with pytest.raises(RuntimeError):
        q.get(timeout=1)
    assert q.get(timeout=1) == "next"


# ---------- concurrency within one process ----------

def test_threads_share_one_queue():
    """Several producer and consumer threads on one queue object (e.g. a free-threaded main process)."""
    q = DejaQueue(200_000)
    n_threads, n_items = 4, 500
    got, lock = [], threading.Lock()

    def produce(k):
        for i in range(n_items):
            q.put((k, i, np.full(8, i)), timeout=5)

    def consume():
        while (item := q.get(timeout=5)) is not None:
            assert np.all(item[2] == item[1])
            with lock:
                got.append(item[:2])

    producers = [threading.Thread(target=produce, args=(k,)) for k in range(n_threads)]
    consumers = [threading.Thread(target=consume) for _ in range(n_threads)]
    for t in producers + consumers:
        t.start()
    for t in producers:
        t.join()
    for _ in consumers:
        q.put(None)
    for t in consumers:
        t.join()
    assert sorted(got) == [(k, i) for k in range(n_threads) for i in range(n_items)]


# ---------- names and lifecycle ----------

@pytest.mark.parametrize("length", [8, 27, 28, 29, 40])
def test_named_queue_resources_are_distinct(length):
    """User-chosen names: everything must stay distinct and fit macOS's ~30-byte POSIX name limits."""
    name = ("q" + os.urandom(16).hex())[:length]
    q = DejaQueue(10_000, name=name)
    sems = [q._put_lock, q._get_lock, q._state_lock, q._items, q._space_gate]
    assert len({s.name for s in sems}) == len(sems), [s.name for s in sems]
    q2 = DejaQueue(name=name, create=False)
    q.put("hello")
    assert q2.get(timeout=1) == "hello"


def test_macos_name_shortening(monkeypatch):
    """The macOS 31-byte workaround, checked on every platform."""
    from dejaq import queues

    monkeypatch.setattr(queues.sys, "platform", "darwin")
    short = "nq1a2b3c4d_Lp"
    assert queues._posix_name(short) == "/" + short  # names that fit are unchanged
    assert queues._shm_name("nq1a2b3c4d_S") == "nq1a2b3c4d_S"
    base = "camera_frames_" + "x" * 20
    names = [queues._posix_name(base + suffix) for suffix in ("_Lp", "_Lg", "_Ls", "_I", "_G")]
    assert len(set(names)) == len(names), names
    for nm in names:
        assert nm.startswith("/") and len(nm.encode()) <= 30  # posix_ipc limit on macOS
        assert queues._posix_name(nm.lstrip("/")) == nm  # fixed point: re-deriving after unpickling
    assert queues._posix_name(base + "_Lp") == names[0]  # deterministic
    shm = [queues._shm_name(base + suffix) for suffix in ("_S", "_B")]
    assert shm[0] != shm[1] and all(not s.startswith("/") and len(s.encode()) <= 30 for s in shm)
    long_unicode = "kamera_ä" * 6
    assert len(queues._posix_name(long_unicode).encode()) <= 30


def test_resources_are_released_when_creator_is_gone():
    name = "q" + os.urandom(6).hex()
    q = DejaQueue(10_000, name=name)
    q.put(1)
    del q
    gc.collect()
    with pytest.raises((FileNotFoundError, OSError)):
        DejaQueue(name=name, create=False)
