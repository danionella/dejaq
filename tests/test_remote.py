import os
import signal
import time
import multiprocessing as mp
import psutil
import pytest
import numpy as np
from dejaq.remote import Actor, ActorDecorator

class Counter:
    def __init__(self, start=0):
        self.value = start
    def inc(self, n=1):
        self.value += n
        return self.value
    def get(self):
        return self.value

def test_actor_basic():
    actor = Actor(Counter, 10)
    assert actor.get() == 10
    assert actor.inc() == 11
    assert actor.inc(5) == 16
    assert actor.get() == 16
    actor.close()

def test_actor_context_manager():
    with Actor(Counter, 100) as actor:
        assert actor.inc() == 101
        assert actor.get() == 101

@ActorDecorator
class Greeter:
    def __init__(self, name):
        self.name = name
    def greet(self):
        return f"Hello, {self.name}!"

def test_actordecorator_basic():
    greeter = Greeter("Alice")
    assert greeter.greet() == "Hello, Alice!"
    greeter.close()


# ---- numpy / deepcopy tests ----

class NumpyActor:
    def __init__(self):
        self._kept = []

    def process(self, arr):
        return float(arr.sum())

    def mutate_then_sum(self, arr):
        # zero the received array; with deepcopy=True this must not affect the caller's copy
        arr[:] = 0
        return float(arr.sum())

    def retain(self, arr):
        self._kept.append(arr)
        return float(arr.sum())


def test_actor_numpy_deepcopy_true_result():
    """deepcopy=True (default): method receives correct data and produces correct result."""
    with Actor(NumpyActor) as a:
        arr = np.ones(100, dtype=np.float32)
        assert a.process(arr) == 100.0
        assert a.process(arr, deepcopy=True) == 100.0


def test_actor_numpy_deepcopy_false_result():
    """deepcopy=False: method produces correct result via zero-copy path."""
    with Actor(NumpyActor) as a:
        arr = np.ones(100, dtype=np.float32)
        assert a.process(arr, deepcopy=False) == 100.0


def test_actor_numpy_deepcopy_true_isolates_args():
    """deepcopy=True: mutations inside the method do not corrupt subsequent calls."""
    with Actor(NumpyActor) as a:
        arr = np.ones(50, dtype=np.float32)
        # mutate_then_sum zeros the array it receives; with deepcopy the next call
        # must still see ones, not zeros.
        assert a.mutate_then_sum(arr, deepcopy=True) == 0.0  # method sees copy → zeroed copy → 0
        assert a.process(arr, deepcopy=True) == 50.0         # next call still sees original ones


def test_actor_numpy_deepcopy_true_no_warning(capfd):
    """deepcopy=True: retaining the array inside the method must NOT emit a RuntimeWarning."""
    with Actor(NumpyActor) as a:
        arr = np.ones(10, dtype=np.float32)
        result = a.retain(arr, deepcopy=True)
        assert result == 10.0
    out = capfd.readouterr()
    assert "RuntimeWarning" not in out.err
    assert "retained" not in out.err


def test_actor_numpy_deepcopy_false_warns_on_retention(capfd):
    """deepcopy=False: retaining a shared-memory array must emit a RuntimeWarning."""
    with Actor(NumpyActor) as a:
        arr = np.ones(10, dtype=np.float32)
        result = a.retain(arr, deepcopy=False)
        assert result == 10.0
    out = capfd.readouterr()
    assert "RuntimeWarning" in out.err and "retain" in out.err


class Storage:
    def __init__(self):
        self.data = None
    def get_data(self):
        return self.data


def test_actor_setattr_numpy_array_is_independent_of_shared_memory(capfd):
    """Setting an actor attribute to a numpy array must store an independent copy.

    If the stored value were a view into the request queue's shared memory,
    subsequent queue activity would silently corrupt the attribute, and the
    SharedMemory destructor in the server would raise
    `BufferError: cannot close exported pointers exist` on shutdown.
    """
    original = (np.arange(100, dtype=np.float32) * 7.0)
    with Actor(Storage) as a:
        a.data = original
        retrieved = a.get_data()
        np.testing.assert_array_equal(retrieved, original)
    # On clean shutdown the server's SharedMemory should release cleanly.
    out = capfd.readouterr()
    assert "BufferError" not in out.err
    assert "cannot close exported pointers" not in out.err


# ---- child-spawning and parent-death tests ----

class Parent:
    """Actor that itself spawns a child Actor during __init__."""
    def __init__(self):
        self._child = Actor(Counter, 0)

    def inc_child(self):
        return self._child.inc()

    def close(self):
        self._child.close()


def test_actor_can_spawn_child_actor():
    """Non-daemon actors must be able to spawn their own child actors."""
    with Actor(Parent) as p:
        assert p.inc_child() == 1
        assert p.inc_child() == 2


def _actor_creator(conn):
    """Spawned by test_parent_death_kills_actor: creates an Actor and reports its PID."""
    a = Actor(Counter, 0)
    conn.send({"pid": a._proc_meta[0]["pid"],
               "ctime": psutil.Process(a._proc_meta[0]["pid"]).create_time()})
    time.sleep(60)  # stay alive until killed


def test_parent_death_kills_actor():
    """When the creating process is SIGKILL-ed, the actor process must also exit."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = ctx.Process(target=_actor_creator, args=(child_conn,), daemon=False)
    p.start()
    child_conn.close()

    info = parent_conn.recv()   # wait for actor to be up
    parent_conn.close()
    actor_pid = info["pid"]
    actor_ctime = info["ctime"]

    os.kill(p.pid, signal.SIGKILL)
    p.join(timeout=5)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            pr = psutil.Process(actor_pid)
            if pr.create_time() != actor_ctime or not pr.is_running():
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    else:
        pytest.fail("actor outlived its creating process by more than 5 seconds")