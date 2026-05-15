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