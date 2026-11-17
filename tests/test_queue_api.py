"""DejaQueue's queue.Queue-compatible interface: arguments, exceptions, convenience methods and shutdown()."""

import queue
import sys
import threading
import time
import warnings
import multiprocessing as mp

import numpy as np
import pytest

import dejaq
from dejaq.queues import DejaQueue, Empty, Full, ShutDown, LegacyDejaQueue


def _fill(q, item=b"x" * 6000):
    q.put(item)  # one item fills more than half of the tiny queues used below


# ---------- exceptions ----------

def test_exceptions_are_standard_queue_exceptions_and_timeouts():
    assert issubclass(Empty, queue.Empty) and issubclass(Empty, TimeoutError)
    assert issubclass(Full, queue.Full) and issubclass(Full, TimeoutError)
    if sys.version_info >= (3, 13):
        assert issubclass(ShutDown, queue.ShutDown)
    assert dejaq.Empty is Empty and dejaq.Full is Full and dejaq.ShutDown is ShutDown


def test_get_timeout_raises_empty():
    q = DejaQueue(10_000)
    t0 = time.monotonic()
    with pytest.raises(queue.Empty):
        q.get(timeout=0.1)
    assert time.monotonic() - t0 >= 0.08
    with pytest.raises(TimeoutError):  # code written for dejaq <= 0.9 keeps working
        q.get(timeout=0)


def test_put_timeout_raises_full():
    q = DejaQueue(10_000)
    _fill(q)
    with pytest.raises(queue.Full):
        q.put(b"y" * 6000, timeout=0.05)
    with pytest.raises(TimeoutError):
        q.put(b"y" * 6000, timeout=0)


# ---------- block / timeout arguments ----------

def _no_warnings(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn()


def test_standard_positional_arguments():
    q = DejaQueue(10_000)
    with pytest.raises(Empty):
        _no_warnings(lambda: q.get(False))  # non-blocking
    t0 = time.monotonic()
    with pytest.raises(Empty):
        _no_warnings(lambda: q.get(True, 0.1))  # block with timeout
    assert time.monotonic() - t0 >= 0.08
    _no_warnings(lambda: q.put(1, True, 1.0))
    _no_warnings(lambda: q.put(2, False))
    assert _no_warnings(lambda: q.get(True)) == 1  # formerly: a 1 s timeout
    assert _no_warnings(lambda: q.get(None)) == 2  # formerly and now: wait indefinitely


def test_block_false_ignores_timeout():
    q = DejaQueue(10_000)
    t0 = time.monotonic()
    with pytest.raises(Empty):
        q.get(block=False, timeout=5)
    _fill(q)
    with pytest.raises(Full):
        q.put(b"y" * 6000, block=False, timeout=5)
    assert time.monotonic() - t0 < 1


def test_positional_timeout_still_works_but_is_deprecated():
    q = DejaQueue(10_000)
    t0 = time.monotonic()
    with pytest.warns(DeprecationWarning, match="timeout=") as rec:
        with pytest.raises(Empty):
            q.get(0.1)
    assert time.monotonic() - t0 >= 0.08
    assert rec[0].filename == __file__  # the warning points at the caller
    _fill(q)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(Full):
            q.put(b"y" * 6000, 0.05)
    with pytest.warns(DeprecationWarning):
        assert q.get(np.float64(1.0)) == b"x" * 6000
    with pytest.raises(TypeError):
        q.get(1.0, 2.0)


def test_peek_and_callback_are_keyword_arguments():
    q = DejaQueue(10_000)
    q.put(np.arange(3))
    assert np.array_equal(q.get(peek_only=True), np.arange(3))
    assert q.get(timeout=1, callback=lambda a: int(a.sum())) == 3


# ---------- convenience methods ----------

def test_nowait_qsize_empty_full():
    q = DejaQueue(10_000)
    assert q.empty() and q.qsize() == 0 and not q.full()
    with pytest.raises(Empty):
        q.get_nowait()
    q.put_nowait("a")
    assert not q.empty() and q.qsize() == 1
    import pickle

    overhead = len(pickle.dumps(b"z" * 5000, protocol=5)) - 5000 + 16  # pickle framing + item header
    big = b"z" * (q.bytes_free - overhead - 10)  # leaves 10 bytes: less than the smallest item needs
    q.put_nowait(big)
    assert q.full() and q.qsize() == 2
    with pytest.raises(Full):
        q.put_nowait(None)
    assert q.get_nowait() == "a" and q.get_nowait() == big
    assert q.empty() and not q.full()


def test_bytes_free_and_former_name():
    q = DejaQueue(10_000)
    assert q.bytes_free == q.bytes_available == q.cap - 1
    q.put(b"x" * 1000)
    assert q.bytes_free == q.bytes_available < q.cap - 1000


# ---------- shutdown ----------

def test_shutdown_lets_consumers_drain_then_raises():
    q = DejaQueue(10_000)
    for i in range(3):
        q.put(i)
    q.shutdown()
    assert q.is_shutdown and q.closed
    with pytest.raises(ShutDown):
        q.put(3)
    assert q.get(peek_only=True) == 0
    assert [q.get(), q.get(), q.get()] == [0, 1, 2]
    with pytest.raises(ShutDown):
        q.get()
    with pytest.raises(ShutDown):
        q.get(block=False)


def test_shutdown_immediate_drops_items():
    q = DejaQueue(10_000)
    for i in range(3):
        q.put(i)
    q.shutdown(immediate=True)
    with pytest.raises(ShutDown):
        q.get()
    assert q.qsize() == 0


def test_shutdown_wakes_all_waiting_consumers_and_producers():
    q = DejaQueue(10_000)
    results = []
    lock = threading.Lock()

    def consume():
        try:
            q.get()
        except ShutDown:
            with lock:
                results.append("consumer")

    consumers = [threading.Thread(target=consume) for _ in range(4)]
    for t in consumers:
        t.start()
    time.sleep(0.3)  # all wait for an item
    q2 = DejaQueue(10_000)
    _fill(q2)

    def produce():
        try:
            q2.put(b"y" * 6000)
        except ShutDown:
            with lock:
                results.append("producer")

    producers = [threading.Thread(target=produce) for _ in range(3)]
    for t in producers:
        t.start()
    time.sleep(0.3)  # all wait for space
    q.shutdown()
    q2.shutdown()
    for t in consumers + producers:
        t.join(5)
    assert not any(t.is_alive() for t in consumers + producers)
    assert sorted(results) == ["consumer"] * 4 + ["producer"] * 3


def _child_iterates(q, out):
    out.put(list(q))  # ends when the queue is shut down and empty


def test_shutdown_reaches_other_processes_and_ends_iteration():
    ctx = mp.get_context("spawn")
    q = DejaQueue(10_000)
    out = ctx.Queue()
    p = ctx.Process(target=_child_iterates, args=(q, out))
    p.start()
    for i in range(5):
        q.put(i)
    q.shutdown()
    assert out.get(timeout=30) == [0, 1, 2, 3, 4]
    p.join(10)
    assert p.exitcode == 0


def test_closed_setter_and_reopen():
    q = DejaQueue(10_000)
    q.closed = True  # former interface: now a shutdown
    with pytest.raises(ShutDown):
        q.put(1)
    q.closed = False  # reopen
    q.put(1)
    assert q.get(timeout=1) == 1


def test_ellipsis_still_ends_iteration():
    q = DejaQueue(10_000)
    q.put(1)
    q._signal_stop()
    assert list(q) == [1]
    assert q.get(timeout=1) is Ellipsis  # put back for other consumers


# ---------- legacy ----------

def test_legacy_queue_is_deprecated():
    with pytest.warns(DeprecationWarning, match="LegacyDejaQueue is deprecated"):
        LegacyDejaQueue(10_000)
