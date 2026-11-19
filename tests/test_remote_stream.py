"""actor.method.stream(queue): a running actor puts the method's results into a queue between calls."""

import threading
import time

import numpy as np
import psutil
import pytest

from dejaq import Actor, DejaQueue, Parallel, ShutDown


class FakeCamera:
    def __init__(self, fail=False):
        self.n, self.running, self.exposure, self.fail, self.size = 0, True, 1.0, fail, 100_000

    def grab_frame(self, value=None):
        if self.fail:
            self.fail = self.fail - 1 if self.fail is not True else True
            raise OSError("camera disconnected")
        if not self.running:
            return None  # nothing to stream now
        time.sleep(0.001)
        self.n += 1
        frame = np.zeros(self.size, np.uint16)
        frame[0] = self.n if value is None else value
        return frame

    def set_exposure(self, ms):
        self.exposure = ms
        return self.n

    def pause(self, paused=True):
        self.running = not paused

    def count(self):
        return self.n

    def fail_next(self, n):
        self.fail = n

    def make_frames(self, size):
        self.size = size

    def grab_unpicklable(self):
        time.sleep(0.001)
        self.n += 1
        return threading.Lock()  # (cannot be pickled)


def test_frames_stream_in_order_and_calls_run_in_between():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera) as cam:
        assert q.qsize() == 0  # (not streaming yet)
        cam.grab_frame.stream(q)
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
    with Actor(FakeCamera) as cam:
        cam.grab_frame.stream(q)
        cam.pause()
        n = cam.count()
        q.purge()
        time.sleep(0.3)
        assert q.qsize() == 0 and cam.count() == n
        cam.pause(False)
        assert int(q.get(timeout=30)[0]) > n


def test_stream_feeds_a_pipeline():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera) as cam:
        cam.grab_frame.stream(q, on_full="drop_oldest")
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
    with Actor(FakeCamera, True) as cam:
        cam.grab_frame.stream(q)
        assert cam.set_exposure(3.0) == 0
        assert q.qsize() == 0


def test_shut_down_stream_stops_streaming_but_not_the_actor():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera) as cam:
        cam.grab_frame.stream(q)
        q.get(timeout=30)
        q.shutdown()
        n = -1
        while cam.count() != n:  # until the actor stopped grabbing
            n = cam.count()
            time.sleep(0.2)
        assert cam.set_exposure(4.0) == n


def test_stream_with_arguments_can_be_replaced_and_stopped():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera) as cam:
        cam.grab_frame.stream(q, 7)
        assert int(q.get(timeout=30)[0]) == 7
        cam.grab_frame.stream(q, value=8)  # replaces the stream
        while int(q.get(timeout=30)[0]) != 8:
            pass
        cam.grab_frame.stream(None)  # stops
        q.purge()
        time.sleep(0.2)
        assert q.qsize() == 0
        assert not q.closed  # (only stopping the actor shuts the queue down)


def test_streaming_resumes_after_a_failure_while_the_actor_is_polled():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera) as cam:
        cam.grab_frame.stream(q)
        cam.fail_next(1)  # one failure: retried after 1 s, although calls keep coming in
        t0, n = time.monotonic(), cam.count()
        while cam.count() == n and time.monotonic() - t0 < 30:
            q.purge()
            time.sleep(0.1)
        assert cam.count() > n


def test_a_result_larger_than_the_queue_stops_streaming():
    q = DejaQueue(10 * 200_100)
    with Actor(FakeCamera) as cam:
        cam.make_frames(2_000_000)  # 4 MB: does not fit
        assert cam.grab_frame.stream(q) is q
        n = -1
        while cam.count() != n:  # streaming stops after the first result
            n = cam.count()
            time.sleep(0.2)
        assert n == 1 and cam.set_exposure(5.0) == 1


def test_replacing_the_queue_leaves_the_old_one_open():
    q1, q2 = DejaQueue(10 * 200_100), DejaQueue(10 * 200_100)
    with Actor(FakeCamera) as cam:
        cam.grab_frame.stream(q1)
        q1.get(timeout=30)
        cam.grab_frame.stream(q2)
        q2.get(timeout=30)
        assert not q1.closed
    assert q2.closed and not q1.closed  # (stopping the actor shuts only the current stream's queue down)


def test_stream_is_checked():
    q = DejaQueue(1_000_000)
    with Actor(FakeCamera) as cam:
        with pytest.raises(ValueError):
            cam.grab_frame.stream("not a queue")
        for mode in ("newest", "block"):  # (the actor must not wait for space)
            with pytest.raises(ValueError):
                cam.grab_frame.stream(q, on_full=mode)
        q.shutdown()
        with pytest.raises(ShutDown):
            cam.grab_frame.stream(q)


def test_an_unpicklable_result_stops_streaming_but_not_the_actor():
    q = DejaQueue(10 * 200_100)
    cam = Actor(FakeCamera)
    cam.grab_unpicklable.stream(q)
    n = -1
    while cam.count() != n:  # streaming stops after the first result
        n = cam.count()
        time.sleep(0.2)
    assert n == 1 and cam.set_exposure(6.0) == 1
    cam.close()


def test_a_failed_stream_keeps_the_current_one():
    q, closed = DejaQueue(10 * 200_100), DejaQueue(1_000_000)
    closed.shutdown()
    with Actor(FakeCamera) as cam:
        cam.grab_frame.stream(q)
        with pytest.raises(ShutDown):
            cam.grab_frame.stream(closed)
        q.purge()
        q.get(timeout=30)  # still streaming into q
    assert q.closed


def test_the_stream_queue_is_shut_down_also_when_the_actor_is_killed():
    q = DejaQueue(10 * 200_100)
    cam = Actor(FakeCamera)
    cam.grab_frame.stream(q)
    q.get(timeout=30)
    psutil.Process(cam._procs[0].pid).kill()
    cam.close()
    assert q.closed  # (by the client: a pipeline reading q finishes)
