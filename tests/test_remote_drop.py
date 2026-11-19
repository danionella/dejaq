"""Actor calls with noreply=True, on_full="drop": skipped when the actor is too busy to take them."""

import time

import numpy as np
import psutil
import pytest

from dejaq import Actor
from dejaq.remote import RemoteError

FRAME = np.zeros(1_000_000, np.uint8)  # 1 MB


class Display:
    def __init__(self):
        self.shown = 0

    def show(self, frame):
        time.sleep(0.02)  # slow
        self.shown += 1

    def count(self):
        return self.shown


def test_dropped_calls_do_not_wait_and_the_rest_arrive():
    with Actor(Display, buffer_bytes=int(3.5 * FRAME.nbytes)) as d:  # room for 3 calls
        d.count()  # (started)
        t0 = time.perf_counter()
        sent = [d.show(FRAME, noreply=True, on_full="drop") for _ in range(100)]
        assert time.perf_counter() - t0 < 1.0  # 100 calls of 20 ms each, but the caller did not wait
        assert 3 <= sum(sent) < 100
        assert d.count() == sum(sent)  # (calls are processed in order: count() runs after all sent ones)
        assert d.show.send(FRAME, on_full="drop") is True


def test_on_full_is_checked():
    with Actor(Display) as d:
        with pytest.raises(ValueError):
            d.show(FRAME, on_full="drop")  # needs noreply=True
        with pytest.raises(ValueError):
            d.show(FRAME, noreply=True, on_full="drop_oldest")  # could discard other callers' requests
        assert d.show(FRAME, noreply=True) is True


def test_dropping_calls_to_a_dead_actor_raises():
    d = Actor(Display, buffer_bytes=int(3.5 * FRAME.nbytes))
    d.count()
    psutil.Process(d._procs[0].pid).kill()
    with pytest.raises(RemoteError):
        for _ in range(10):  # the queue fills up, then a dropped call finds the actor dead
            d.show(FRAME, noreply=True, on_full="drop")
    d.close()


def test_dropped_calls_are_not_logged(caplog):
    with Actor(Display, buffer_bytes=int(3.5 * FRAME.nbytes)) as d:
        d.count()
        with caplog.at_level("WARNING", logger="dejaq.queues"):
            sent = [d.show(FRAME, noreply=True, on_full="drop") for _ in range(50)]
        assert not all(sent) and "dropped" not in caplog.text
