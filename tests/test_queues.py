import pytest
import numpy as np
import hashlib
import random
import multiprocessing as mp
import time

from dejaq.queues import LegacyDejaQueue, DejaQueue, NamedSemaphore

def sema_acquire(name, result_list):
    sem = NamedSemaphore(name=name, create=False)
    acquired = sem.acquire(timeout=2)
    result_list.append(acquired)
    if acquired:
        sem.release()

def test_namedsemaphore_across_processes():
    name = "test_sema_" + str(time.time())
    sem = NamedSemaphore(name=name, create=True, initial=0)
    manager = mp.Manager()
    results = manager.list()

    # Start a process that will try to acquire the semaphore (should block until released)
    p = mp.Process(target=sema_acquire, args=(name, results))
    p.start()
    time.sleep(1)  # Ensure the process is waiting on acquire

    # Now release the semaphore from the main process
    sem.release()
    p.join(timeout=5)

    assert list(results) == [True], f"Semaphore was not acquired in child process: {results}"


def producer(q, items, delay=0, stop=True):
    for item in items:
        q.put(item)
        if delay:
            time.sleep(delay)
    if stop: 
        q._signal_stop()

def consumer(q, results, delay=0, hash=True):
    for item in q:
        item = hash_it(item) if hash else item
        results.append(item)
        if delay:
            time.sleep(delay)

def hash_it(item):
    if isinstance(item, list):
        return [hash_it(i) for i in item]
    import hashlib
    if isinstance(item, (bytes, bytearray)):
        return hashlib.sha256(item).hexdigest()
    elif hasattr(item, "tobytes"):
        return hashlib.sha256(item.tobytes()).hexdigest()
    else:
        return hashlib.sha256(repr(item).encode()).hexdigest()

@pytest.mark.parametrize("QueueCls", [LegacyDejaQueue, DejaQueue])
def test_queue_mp_basic(QueueCls):
    q = QueueCls(1024 * 1024)
    items = [b"foo", b"bar", b"baz", b"qux"]
    manager = mp.Manager()
    results = manager.list()
    p1 = mp.Process(target=producer, args=(q, items))
    p2 = mp.Process(target=consumer, args=(q, results))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    assert list(results) == hash_it(items)

@pytest.mark.parametrize("QueueCls", [LegacyDejaQueue, DejaQueue])
def test_queue_mp_numpy_arrays(QueueCls):
    q = QueueCls(2 * 1024 * 1024)
    arrays = [np.random.randn(1000) for _ in range(5)]
    arrays_bytes = [a.tobytes() for a in arrays]
    manager = mp.Manager()
    results = manager.list()
    p1 = mp.Process(target=producer, args=(q, arrays))
    p2 = mp.Process(target=consumer, args=(q, results))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    for orig, got in zip(arrays_bytes, results):
        assert hash_it(orig) == got

@pytest.mark.parametrize("QueueCls", [LegacyDejaQueue, DejaQueue])
def test_queue_mp_race_condition(QueueCls):
    q = QueueCls(4 * 1024 * 1024)
    N = 50
    arrays = [np.random.bytes(random.randint(100, 10000)) for _ in range(N)]
    hashes = hash_it(arrays)
    manager = mp.Manager()
    results = manager.list()
    p1 = mp.Process(target=producer, args=(q, arrays))
    p2 = mp.Process(target=consumer, args=(q, results))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    assert hashes == list(results)

@pytest.mark.parametrize("QueueCls", [LegacyDejaQueue, DejaQueue])
def test_queue_mp_multiple_producers_consumers(QueueCls):
    q = QueueCls(4 * 1024 * 1024)
    N = 100
    items = [f"item_{i}".encode() for i in range(N)]
    manager = mp.Manager()
    results = manager.list()
    items1 = items[:N//2]
    items2 = items[N//2:]
    p1 = mp.Process(target=producer, args=(q, items1, 0, False))
    p2 = mp.Process(target=producer, args=(q, items2, 0, False))
    c1 = mp.Process(target=consumer, args=(q, results, 0, False)) 
    p1.start()
    p2.start()
    c1.start()
    time.sleep(1)
    q._signal_stop()  # signal stop after both producers are done
    p1.join()
    p2.join()
    c1.join()
    assert len(results) == len(items)
    assert sorted(list(results)) == sorted(items)

@pytest.mark.parametrize("QueueCls", [LegacyDejaQueue, DejaQueue])
def test_queue_mp_stress(QueueCls):
    q = QueueCls(8 * 1024 * 1024)
    N = 500
    items = [np.random.bytes(random.randint(100, 10000)) for _ in range(N)]
    manager = mp.Manager()
    results = manager.list()
    prods = [mp.Process(target=producer, args=(q, items[i::4], 0, False)) for i in range(4)]
    cons_ = [mp.Process(target=consumer, args=(q, results)) for _ in range(4)]
    for p in prods + cons_:
        p.start()
    time.sleep(1)
    q._signal_stop()
    for p in prods + cons_:
        p.join()
    assert len(results) == len(items)
    assert sorted(list(results)) == sorted(hash_it(items))

def test_picklabledejaqueue_mp_objects():
    q = DejaQueue(2 * 1024 * 1024)
    objs = [
        123,
        "hello",
        [1, 2, 3],
        {"a": 1, "b": [2, 3]},
        np.arange(10),
    ]
    manager = mp.Manager()
    results = manager.list()
    p1 = mp.Process(target=producer, args=(q, objs))
    p2 = mp.Process(target=consumer, args=(q, results))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    for orig, got in zip(objs, results):
        assert hash_it(orig) == got


def test_picklabledejaqueue_peek_only_complex():
    q = DejaQueue(1024 * 1024)
    items = [{"x": np.arange(10), "y": "foo"}, [1, 2, 3, {"bar": 99}], b"bytes", np.random.randn(100), "string"]
    for item in items:
        q.put(item)
    # Peek at the first item multiple times
    for _ in range(3):
        peeked = q.get(peek_only=True)
        assert (
            np.all(peeked["x"] == items[0]["x"]) if isinstance(peeked, dict) and "x" in peeked else peeked == items[0]
        )
    # Now consume all items and check order
    for orig in items:
        got = q.get()
        if isinstance(orig, np.ndarray):
            assert np.allclose(got, orig)
        elif isinstance(orig, dict) and "x" in orig:
            assert np.all(got["x"] == orig["x"]) and got["y"] == orig["y"]
        else:
            assert got == orig
