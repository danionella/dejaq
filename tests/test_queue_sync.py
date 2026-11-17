"""Synchronization, naming and lifecycle tests for DejaQueue/NamedSemaphore.

Covers: unique and collision-safe names, get() timeouts with competing consumers, the waiting-producer
protocol (no unbounded semaphore growth, no lost wake-ups), purge() racing with consumers, 64-bit item
headers, and the state-layout version check.
"""

import os
import subprocess
import sys
import threading
import time
import multiprocessing as mp

import numpy as np
import pytest

from dejaq import queues
from dejaq.queues import _IS_WIN, DejaQueue, NamedSemaphore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _drain(sem) -> int:
    """Take all counts of a semaphore without blocking; return how many there were."""
    n = 0
    while sem.acquire(timeout=0):
        n += 1
    return n


def _assert_quiescent(q):
    """Invariants of an idle queue: no registered producers, no stray wake-ups, item count matches."""
    with q._state_lock:
        waiters = int(q._state[queues._WAITERS])
        nitems = int(q._state[queues._NITEMS])
    assert waiters == 0
    assert _drain(q._space_gate) == 0
    claimable = _drain(q._items)
    q._items.release(claimable) if claimable else None
    # one count per item, plus at most a few spare ones (a waiting consumer can hand back its count just
    # after another consumer took the item without one; spare counts are dropped by the next get)
    assert nitems <= claimable <= nitems + 4


# ---------- names (unique, never shared by accident) ----------

def test_random_semaphore_names_are_unique():
    sems = [NamedSemaphore() for _ in range(3000)]
    assert len({s.name for s in sems}) == len(sems)


def test_random_queue_names_are_unique():
    qs = [DejaQueue(4096) for _ in range(200)]
    assert len({q._base for q in qs}) == len(qs)


def test_semaphore_name_collision_is_retried(monkeypatch):
    taken = NamedSemaphore()
    fresh = queues._safe_base("ns")
    names = iter([taken.name.lstrip("/"), fresh])
    monkeypatch.setattr(queues, "_safe_base", lambda prefix="ns": next(names))
    s = NamedSemaphore()
    assert s.name != taken.name
    s.release()
    assert not taken.acquire(timeout=0)  # the two are independent


def test_queue_name_collision_is_retried(monkeypatch):
    taken = DejaQueue(4096)
    names = iter([taken._base, queues._safe_base("nq")])
    monkeypatch.setattr(queues, "_safe_base", lambda prefix="nq": next(names))
    q = DejaQueue(4096)
    assert q._base != taken._base
    q.put("new")
    taken.put("taken")
    assert q.get(timeout=1) == "new" and taken.get(timeout=1) == "taken"


def test_explicit_semaphore_name_that_exists_raises():
    name = "dq_test_" + os.urandom(5).hex()
    a = NamedSemaphore(name, initial=0)
    with pytest.raises(FileExistsError):
        NamedSemaphore(name, initial=5)
    assert not a.acquire(timeout=0)  # untouched


def test_explicit_queue_name_that_exists_raises_and_keeps_the_queue():
    name = "dq_test_" + os.urandom(5).hex()
    q = DejaQueue(4096, name=name)
    q.put("kept")
    with pytest.raises(FileExistsError):
        DejaQueue(4096, name=name)
    # the failed attempt must not have unlinked or reset anything of the existing queue
    assert DejaQueue(name=name, create=False).get(timeout=1) == "kept"


def test_failed_creation_cleans_up(monkeypatch):
    """If creating a queue fails halfway, what was created is removed again and the name is reusable."""
    name = "dq_test_" + os.urandom(5).hex()
    real = queues.NamedSemaphore
    calls = []

    def flaky(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:  # the second of the queue's two plain semaphores
            raise OSError("simulated failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(queues, "NamedSemaphore", flaky)
    with pytest.raises(OSError, match="simulated"):
        DejaQueue(4096, name=name)
    monkeypatch.setattr(queues, "NamedSemaphore", real)
    q = DejaQueue(4096, name=name)  # FileExistsError if anything was left behind
    q.put(1)
    assert q.get(timeout=1) == 1


@pytest.mark.skipif(_IS_WIN, reason="Windows semaphores vanish with their last handle")
def test_stale_objects_of_a_crashed_queue():
    """Lock files left by a crashed creator (macOS keeps them) are reused; semaphores left behind (only if its
    resource tracker died too) make creating a queue of the same name fail cleanly."""
    import posix_ipc

    name = "dq_test_" + os.urandom(5).hex()
    _, _, lock_names, sem_names = queues.NamedByteRing._names(name)
    for nm in lock_names:
        lk = queues._RobustLock(nm, auto_unlink=False)
        lk._finalizer.detach()  # leave the file behind, as a crashed process would
        lk.close()
    q = DejaQueue(4096, name=name)
    q.put("fresh")
    assert q.get(timeout=1) == "fresh"
    _assert_quiescent(q)
    del q
    name2 = "dq_test_" + os.urandom(5).hex()
    stale = posix_ipc.Semaphore(queues._posix_name(queues.NamedByteRing._names(name2)[3][1]), posix_ipc.O_CREX)
    try:
        with pytest.raises(FileExistsError):
            DejaQueue(4096, name=name2)
        DejaQueue(4096, name=name2 + "b")  # (nothing else was affected)
    finally:
        stale.close()
        stale.unlink()


# ---------- get(timeout) with competing consumers ----------

def test_get_timeout_is_honored_while_another_consumer_waits():
    q = DejaQueue(10_000)
    got = []
    blocked = threading.Thread(target=lambda: got.append(q.get()), daemon=True)
    blocked.start()
    time.sleep(0.2)  # the first consumer now waits for an item
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        q.get(timeout=0.3)
    assert 0.25 <= time.monotonic() - t0 < 2.0
    q.put("a")
    q.put("b")
    assert q.get(timeout=2) in ("a", "b")
    blocked.join(2)
    assert got and got[0] in ("a", "b")
    _assert_quiescent(q)


def _blocking_get(q, out):
    out.put(q.get(timeout=20))


def test_get_timeout_is_honored_while_another_process_waits():
    ctx = mp.get_context("spawn")
    q = DejaQueue(10_000)
    out = ctx.Queue()
    p = ctx.Process(target=_blocking_get, args=(q, out))
    p.start()
    try:
        time.sleep(1.0)  # child started and is waiting for an item
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            q.get(timeout=0.3)
        assert time.monotonic() - t0 < 2.0
        q.put("for the child")
        assert out.get(timeout=10) == "for the child"
    finally:
        p.join(10)
    assert p.exitcode == 0


def test_get_nonblocking_returns_available_item_despite_waiting_consumer():
    q = DejaQueue(10_000)
    got = []
    t = threading.Thread(target=lambda: got.append(q.get(timeout=2)), daemon=True)
    t.start()
    time.sleep(0.1)  # the thread waits for an item
    q.put(1)
    q.put(2)
    mine = q.get(timeout=0)  # an item is there: must not time out, whatever the waiting consumer does
    t.join(2)
    assert sorted(got + [mine]) == [1, 2]
    _assert_quiescent(q)


def test_lost_item_counts_do_not_block_consumers():
    """Counts lost with a process that died (or spare ones) must not hide or invent items."""
    q = DejaQueue(10_000)
    for i in range(3):
        q.put(i)
    assert _drain(q._items) == 3  # as if three consumers had died holding counts
    assert [q.get(timeout=1) for _ in range(3)] == [0, 1, 2]
    q._items.release(2)  # spare counts without items
    with pytest.raises(TimeoutError):
        q.get(timeout=0.1)
    q.put("x")
    assert q.get(timeout=1) == "x"
    _assert_quiescent(q)


# ---------- purge ----------

def test_purge_does_not_deadlock_with_waiting_consumer():
    q = DejaQueue(10_000)
    got = []
    t = threading.Thread(target=lambda: got.append(q.get(timeout=5)), daemon=True)
    t.start()
    time.sleep(0.2)
    done = threading.Event()
    threading.Thread(target=lambda: (q.purge(), done.set()), daemon=True).start()
    assert done.wait(2), "purge() blocked behind a waiting consumer"
    q.put("after purge")  # the waiting consumer still works
    t.join(5)
    assert got == ["after purge"]


def test_consumer_that_claimed_a_purged_item_waits_for_the_next():
    q = DejaQueue(10_000)
    q.put("purged")
    got = []
    q._get_lock.acquire()  # hold the consumer back after it has claimed the item
    c = threading.Thread(target=lambda: got.append(q.get(timeout=5)), daemon=True)
    c.start()
    time.sleep(0.3)
    with q._put_lock:
        q._purge_locked()  # what purge() does, while the consumer holds a claim on the old item
    q._get_lock.release()
    time.sleep(0.3)
    assert not got  # it did not read the purged item (or garbage)
    q.put("next")
    c.join(5)
    assert got == ["next"]
    _assert_quiescent(q)


def test_purge_wakes_waiting_producer():
    q = DejaQueue(10_000)
    q.put(b"x" * 6000)
    done = threading.Event()
    threading.Thread(target=lambda: (q.put(b"y" * 6000, timeout=5), done.set()), daemon=True).start()
    time.sleep(0.2)
    q.purge()
    assert done.wait(2)
    assert q.get(timeout=1) == b"y" * 6000
    _assert_quiescent(q)


# ---------- waiting producers ----------

def test_space_gate_does_not_accumulate_counts():
    """Formerly every get() released the gate, growing it without bound (overflow after ~2**31 items)."""
    q = DejaQueue(100_000)
    for i in range(20_000):
        q.put(i)
        assert q.get() == i
    _assert_quiescent(q)


def test_many_blocked_producers_are_all_woken():
    q = DejaQueue(5_000)
    n_prod, n_items = 6, 40
    got = []

    def produce(k):
        for i in range(n_items):
            q.put((k, i, b"p" * 1500), timeout=20)

    producers = [threading.Thread(target=produce, args=(k,)) for k in range(n_prod)]
    for t in producers:
        t.start()
    while len(got) < n_prod * n_items:
        got.append(q.get(timeout=20)[:2])
        if len(got) % 7 == 0:
            time.sleep(0.005)  # let producers pile up on the full queue
    for t in producers:
        t.join(5)
    assert sorted(got) == [(k, i) for k in range(n_prod) for i in range(n_items)]
    _assert_quiescent(q)


def test_producer_interrupted_while_waiting_deregisters(monkeypatch):
    q = DejaQueue(10_000)
    q.put(b"x" * 6000)

    def interrupted(timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(q._space_gate, "acquire", interrupted)
    with pytest.raises(KeyboardInterrupt):
        q.put(b"y" * 6000)
    monkeypatch.undo()
    q.get(timeout=1)
    _assert_quiescent(q)


def _timeout_storm(q, k, n, seed):
    """Producer that puts with short timeouts and retries: deregistration races with consumers' wake-ups."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        item = (k, i, np.full(int(rng.integers(0, 3000)), i % 256, dtype=np.uint8))
        while True:
            try:
                q.put(item, timeout=float(rng.choice([0.0, 0.0005, 0.002, 0.05])))
                break
            except TimeoutError:
                pass


def _consume_until_none(q, n_none_expected, out, seed):
    rng = np.random.default_rng(seed)
    seen = []
    while True:
        try:
            peek = bool(rng.random() < 0.05)
            item = q.get(timeout=float(rng.choice([0.0, 0.001, 0.05])), peek_only=peek)
        except TimeoutError:
            continue
        if peek:
            continue
        if item is None:
            break
        k, i, arr = item
        assert np.all(arr == i % 256)
        seen.append((k, i))
    out.put(seen)


@pytest.mark.parametrize("use_processes", [False, True], ids=["threads", "processes"])
def test_stress_timeouts_peeks_and_wraparound(use_processes):
    """Several producers and consumers on a tiny ring, all using short timeouts (and some peeks).

    Every item must arrive exactly once and intact, and the queue must end in a consistent idle state.
    """
    n_prod, n_cons, n_items = 3, 3, 150 if use_processes else 300
    q = DejaQueue(12_000)
    if use_processes:
        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        Worker = ctx.Process
    else:
        import queue as _q

        out = _q.Queue()
        Worker = threading.Thread
    producers = [Worker(target=_timeout_storm, args=(q, k, n_items, k)) for k in range(n_prod)]
    consumers = [Worker(target=_consume_until_none, args=(q, n_prod, out, 100 + c)) for c in range(n_cons)]
    for w in producers + consumers:
        w.start()
    for w in producers:
        w.join(120)
    for _ in consumers:
        q.put(None, timeout=30)
    seen = [x for _ in consumers for x in out.get(timeout=120)]
    for w in consumers:
        w.join(30)
    assert sorted(seen) == [(k, i) for k in range(n_prod) for i in range(n_items)]
    _assert_quiescent(q)
    assert q.nitems == 0 and q.is_empty


# ---------- item format and versions ----------

def test_state_slot_numbers():
    """put()/get()/_finish_get() use literal slot numbers on their hot paths; they must match the names."""
    assert (queues._HEAD, queues._TAIL, queues._NITEMS, queues._WAITERS) == (0, 1, 2, 5)


def test_header_lengths_are_64_bit():
    import struct

    lens = [5, 2**32 + 7, 2**40]
    hdr = queues._pack_header(lens)
    (k,) = struct.unpack("<Q", hdr[:8])
    assert k == 3 and list(struct.unpack(f"<{k}Q", hdr[8:])) == lens


@pytest.mark.skipif(os.environ.get("DEJAQ_TEST_LARGE") != "1", reason="needs ~13 GB of RAM; set DEJAQ_TEST_LARGE=1")
def test_item_larger_than_4GiB():
    n = 2**32 + 4096
    q = DejaQueue(n + 2**20)
    a = np.zeros(n, dtype=np.uint8)
    a[[0, 2**32 - 1, 2**32, n - 1]] = (1, 2, 3, 4)
    q.put(a)
    del a
    b = q.get(timeout=60)
    assert b.shape == (n,) and tuple(b[[0, 2**32 - 1, 2**32, n - 1]]) == (1, 2, 3, 4)


def test_attaching_to_an_incompatible_queue_fails_clearly():
    q = DejaQueue(4096)
    with q._state_lock:
        q._state[queues._MAGIC] = 0
    try:
        with pytest.raises(RuntimeError, match="incompatible dejaq version"):
            DejaQueue(name=q._base, create=False)
    finally:
        with q._state_lock:
            q._state[queues._MAGIC] = queues._STATE_MAGIC


def test_attaching_to_a_previous_version_queue_fails_clearly():
    """Queues of dejaq <= 0.8 have a 5-slot (40 byte) state block."""
    from multiprocessing import shared_memory

    base = "dq_test_" + os.urandom(5).hex()
    st_name, buf_name, _, _ = queues.NamedByteRing._names(base)
    old = shared_memory.SharedMemory(name=st_name, create=True, size=40)
    old.buf.cast("q")[4] = 4096  # cap
    try:
        with pytest.raises(RuntimeError, match="incompatible dejaq version"):
            DejaQueue(name=base, create=False)
    finally:
        old.close()
        old.unlink()


# ---------- process lifecycle ----------

_FORK_CHILD = """
import atexit, os
from dejaq.queues import DejaQueue
q = DejaQueue(4096)
q.put("still here")
pid = os.fork()
if pid == 0:  # child: run exit handlers (weakref finalizers) as a normal interpreter exit would
    try:
        atexit._run_exitfuncs()
    finally:
        os._exit(0)
os.waitpid(pid, 0)
q2 = DejaQueue(name=q._base, create=False)  # FileNotFoundError if the child unlinked it
assert q2.get(timeout=1) == "still here"
q2.put(1)  # semaphores are intact too
assert q.get(timeout=1) == 1
print("ok")
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_does_not_unlink_the_parents_queue():
    """A forked child inherits the creator's objects; its exit handlers must not remove the queue.

    Runs in a fresh single-threaded interpreter, since forking pytest's process (with threads) is unsafe.
    """
    r = _run_python(_FORK_CHILD)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def _run_python(code):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env)
