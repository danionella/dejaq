"""put(on_full=...): what to do when the queue is full, and stage.qsize()."""

import multiprocessing as mp
import threading
import time

import numpy as np
import pytest

from dejaq import DejaQueue, Full, Parallel, ShutDown

FRAME = np.zeros(100_000, np.uint8)  # ~100 kB items: a 1 MB queue holds 9 of them


def _fill(q, n, **kw):
    out = []
    for i in range(n):
        f = FRAME.copy()
        f[0] = i
        out.append(q.put(f, **kw))
    return out


def _drain(q):
    return [int(q.get()[0]) for _ in range(q.qsize())]


def test_drop_discards_the_new_item():
    q = DejaQueue(1_000_000)
    assert _fill(q, 20, on_full="drop") == [True] * 9 + [False] * 11
    assert q.dropped == 11 and _drain(q) == list(range(9))


def test_drop_oldest_keeps_the_newest_items_in_order():
    q = DejaQueue(1_000_000)
    assert all(_fill(q, 20, on_full="drop_oldest"))
    assert q.dropped == 11 and _drain(q) == list(range(11, 20))


def test_flush_discards_everything_queued():
    q = DejaQueue(1_000_000)
    _fill(q, 20, on_full="flush")
    assert q.dropped == 18 and _drain(q) == [18, 19]  # flushed at items 9 and 18


def test_keep_only_the_latest():
    q = DejaQueue(int(1.5 * FRAME.nbytes))  # room for one item
    _fill(q, 5, on_full="drop_oldest")
    assert q.qsize() == 1 and int(q.get()[0]) == 4


def test_block_is_the_default_and_on_full_is_checked():
    q = DejaQueue(1_000_000)
    _fill(q, 9)
    with pytest.raises(Full):
        q.put(FRAME, timeout=0.05)
    with pytest.raises(ValueError):
        q.put(FRAME, on_full="newest")
    q.shutdown()
    with pytest.raises(ShutDown):
        q.put(FRAME, on_full="drop_oldest")


class _NotImportable:
    def __reduce__(self):
        return (_missing, ())  # unpickling fails: dropped items are skipped, not unpickled


_unpickled = []


def _missing():
    _unpickled.append(1)
    raise RuntimeError("unpickled")


def test_dropped_items_are_not_unpickled():
    q = DejaQueue(1_000_000)
    q.put(_NotImportable())
    _fill(q, 10, on_full="drop_oldest")  # drops it and item 0
    assert q.dropped == 2 and _drain(q) == list(range(1, 10)) and not _unpickled


def test_blocked_producer_is_woken_by_a_dropping_one():
    q = DejaQueue(1_000_000)
    _fill(q, 9)
    t = threading.Thread(target=q.put, args=(FRAME,), kwargs={"timeout": 10})  # waits for space
    t.start()
    time.sleep(0.2)
    q.put(FRAME, on_full="flush")  # frees all space: the waiting producer must get in
    t.join(10)
    assert not t.is_alive() and q.qsize() == 2


def test_waiting_consumer_and_producer_counts_stay_consistent():
    q = DejaQueue(1_000_000)
    got = []
    t = threading.Thread(target=lambda: got.extend(int(x[0]) for x in q))  # waits on the empty queue
    t.start()
    time.sleep(0.1)
    _fill(q, 200, on_full="drop_oldest")
    q.shutdown()
    t.join(10)
    assert not t.is_alive()
    assert got == sorted(got) and got[-1] == 199 and len(got) + q.dropped == 200


def _camera(q, n):
    frame = np.zeros(4_000_000, np.uint8)
    for i in range(n):
        frame[:8] = np.frombuffer(np.int64(i).tobytes(), np.uint8)
        q.put(frame, on_full="drop_oldest")
    q.put(None)
    q.put(q.dropped)


def test_slow_consumer_in_another_process_gets_the_newest_frames_in_order():
    q = DejaQueue(10 * 4_100_000)
    p = mp.get_context("spawn").Process(target=_camera, args=(q, 300))
    p.start()
    seen = []
    while (frame := q.get(timeout=60)) is not None:
        seen.append(int(frame[:8].view(np.int64)[0]))
        time.sleep(0.005)
    dropped = q.get(timeout=10)
    p.join(10)
    assert p.exitcode == 0
    assert seen == sorted(seen) and seen[-1] == 299 and len(seen) + dropped == 300


@pytest.mark.parametrize("n_workers, ordered", [(1, True), (3, True), (3, False)])
def test_stage_qsize(n_workers, ordered):
    s = Parallel(n_workers=n_workers, ordered=ordered)(abs)(range(5))
    s.start()
    deadline = time.monotonic() + 30
    while not all(s._done) and time.monotonic() < deadline:  # all results (and end markers) are in
        time.sleep(0.05)
    assert s.qsize() == 5
    assert sorted(s) == list(range(5)) and s.qsize() == 0
