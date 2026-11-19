"""Actor(..., stream=(method, queue)): the actor puts the method's results into a queue between calls."""

import time

import numpy as np
import pytest

from dejaq import Actor, DejaQueue, Parallel


class FakeCamera:
    def __init__(self, fail=False):
        self.n, self.running, self.exposure, self.fail = 0, True, 1.0, fail

    def grab_frame(self):
        if self.fail:
            raise OSError("camera disconnected")
        if not self.running:
            return None  # nothing to stream now
        time.sleep(0.001)
        self.n += 1
        frame = np.zeros(100_000, np.uint16)
        frame[0] = self.n
        return frame

    def set_exposure(self, ms):
        self.exposure = ms
        return self.n

    def pause(self, paused=True):
        self.running = not paused

    def count(self):
        return self.n


def test_frames_stream_in_order_and_calls_run_in_between():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera, stream=("grab_frame", q)) as cam:
        got = [int(q.get(timeout=30)[0]) for _ in range(50)]
        t0 = time.perf_counter()
        assert cam.set_exposure(2.0) > 0  # served between two frames
        assert time.perf_counter() - t0 < 1.0
        assert cam.exposure == 2.0
    assert got == sorted(got) and len(set(got)) == 50
    rest = list(q)  # the actor shut the queue down when it stopped: iteration ends
    assert all(int(f[0]) > got[-1] for f in rest)


def test_none_streams_nothing():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera, stream=("grab_frame", q)) as cam:
        cam.pause()
        n = cam.count()
        q.purge()
        time.sleep(0.3)
        assert q.qsize() == 0 and cam.count() == n
        cam.pause(False)
        assert int(q.get(timeout=30)[0]) > n


def test_stream_feeds_a_pipeline():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera, stream=("grab_frame", q, "drop_oldest")):
        stage = Parallel(n_workers=2)(np.max)(q)
        out = []
        for x in stage:
            out.append(int(x))
            if len(out) == 30:
                break
        stage.close()
    assert out == sorted(out)


def test_failing_stream_method_keeps_the_actor_responsive():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera, True, stream=("grab_frame", q)) as cam:
        assert cam.set_exposure(3.0) == 0
        assert q.qsize() == 0


def test_shut_down_stream_stops_streaming_but_not_the_actor():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera, stream=("grab_frame", q)) as cam:
        q.get(timeout=30)
        q.shutdown()
        time.sleep(0.2)
        n = cam.count()
        time.sleep(0.2)
        assert cam.count() == n  # no more grabs
        assert cam.set_exposure(4.0) == n


def test_stream_is_checked():
    q = DejaQueue(1_000_000)
    for bad in ("grab_frame", ("grab_frame",), ("grab_frame", q, "newest"), (1, q)):
        with pytest.raises(ValueError):
            Actor(FakeCamera, stream=bad)
