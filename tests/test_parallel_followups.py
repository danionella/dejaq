"""Tests for tee(), closing a pipeline, and worker processes that are left behind."""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

from dejaq import Parallel, parallel
from dejaq.parallel import lazymap
from _testutil import REPO_ROOT, all_gone, run_script


def _remove_lock_files_of_killed_creators(output):
    """macOS: the lock files of queues whose creating process was killed remain (empty files in /tmp; on Linux
    the resource tracker removes them). The scripts print these queues' names as 'queues <name> ...'."""
    if sys.platform != "darwin":
        return
    from dejaq import queues
    for line in output.splitlines():
        if line.startswith("queues"):
            for base in line.split()[1:]:
                for nm in queues.NamedByteRing._names(base)[2]:
                    try:
                        os.unlink(queues._RobustLock._path_for(nm))
                    except FileNotFoundError:
                        pass


def add5(x):
    return x + 5


def ident(x):
    return x


def big(x):
    return np.ones(500_000)  # 4 MB


def fails_at_3(x):
    if x == 3:
        raise ValueError("bad item 3")
    return x


# ---------- tee ----------

def test_tee_of_a_temporary_stage():
    """The branches keep the stage they split alive (it used to be garbage-collected at once)."""
    a, b = Parallel(n_workers=2)(add5)(range(10)).tee(2)
    assert [(x, y) for x, y in zip(a, b)] == [(i + 5, i + 5) for i in range(10)]


def test_tee_upstream_error_reaches_every_branch():
    a, b = Parallel(n_workers=2)(fails_at_3)(range(10)).tee(2)
    with pytest.raises(ValueError, match="bad item 3"):
        for _ in zip(a, b):
            pass
    with pytest.raises(ValueError, match="bad item 3"):
        list(b)


def big_12mb(x):
    return np.ones(1_500_000)


def test_tee_branches_take_the_queue_size_of_the_stage():
    up = Parallel(n_workers=1, buffer_bytes=40e6)(big_12mb)(range(3))  # larger than the default 10 MB
    a, b = up.tee(2)
    assert [x.shape for x, _ in zip(a, b)] == [(1_500_000,)] * 3


def test_tee_of_a_tee_and_of_a_stage_with_a_cloudpickle_input():
    a, b = Parallel(n_workers=1)(ident)(range(5)).tee(2)
    a1, a2 = a.tee(2)
    assert [t for t in zip(a1, a2, b)] == [(i, i, i) for i in range(5)]
    c, d = Parallel(n_workers=1)(ident)(map(lambda x: 2 * x, range(3))).tee(2)
    assert list(zip(c, d)) == [(0, 0), (2, 2), (4, 4)]


def test_tee_branches_feed_further_stages():
    a, b = Parallel(n_workers=2)(add5)(range(20)).tee(2)
    pa = Parallel(n_workers=2)(add5)(a)
    pb = Parallel(n_workers=3)(ident)(b)
    assert list(zip(pa, pb)) == [(i + 10, i + 5) for i in range(20)]


def test_tee_closing_one_branch_keeps_the_others_going():
    up = Parallel(n_workers=2)(big)(range(20))
    a, b = up.tee(2)
    next(iter(a))
    a.close()
    assert len(list(b)) == 20  # would block once a's queue is full if a were still served
    b.close()
    assert all_gone(up.pids + up._it.pids + a.pids)


def test_tee_garbage_collected_branches_stop_the_tee_and_the_stage():
    import gc
    up = Parallel(n_workers=2)(big)(range(20))
    pids = up.pids + up._it.pids
    a, b = up.tee(2)
    pids += a.pids
    next(iter(a))
    del up, a, b
    gc.collect()
    assert all_gone(pids)


# ---------- closing a pipeline ----------

def test_closing_the_last_stage_closes_the_stages_that_feed_it():
    stage1 = Parallel(n_workers=2)(big)(range(100))
    stage2 = Parallel(n_workers=2)(ident)(stage1)
    stage3 = Parallel(n_workers=1)(ident)(stage2)
    next(iter(stage3))
    stage3.close()
    assert all_gone(stage1.pids + stage1._it.pids + stage2.pids + stage3.pids)
    with pytest.raises(RuntimeError, match="closed"):
        list(stage1)


def test_an_error_closes_the_whole_pipeline():
    stage1 = Parallel(n_workers=2)(fails_at_3)(range(100))
    stage2 = Parallel(n_workers=2)(ident)(stage1)
    with pytest.raises(ValueError):
        stage2.run(progress=False)
    assert all_gone(stage1.pids + stage1._it.pids + stage2.pids)


def test_closing_a_zip_closes_its_inputs():
    a = Parallel(n_workers=2)(big)(range(100))
    b = Parallel(n_workers=1)(big)(range(100))
    z = parallel.zip(a, b)
    next(iter(z))
    z.close()
    assert all_gone(a.pids + a._it.pids + b.pids + z.pids)


# ---------- processes left behind ----------

_NESTED = """
import os, sys, time
import numpy as np
from dejaq import Parallel

def inner(x):
    return np.ones(500_000)

def outer(x):
    stage = Parallel(n_workers=2)(inner)(range(100))  # a stage created in a worker
    next(iter(stage))
    print("grandchildren", *stage.pids, flush=True)
    print("queues", stage._out_queue._base, stage._it._out_queue._base, flush=True)
    time.sleep(60)
    return x

if __name__ == "__main__":
    stage = Parallel(n_workers=1)(outer)(range(10))
    it = iter(stage)
    time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0)
    stage.close()
"""


def test_closing_a_stage_also_ends_the_processes_its_workers_started(tmp_path):
    code = _NESTED.replace("    it = iter(stage)\n", "    it = iter(stage)\n    time.sleep(5)\n")
    r = run_script(tmp_path, code)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("grandchildren")]
    assert line, (r.stdout, r.stderr[-3000:])
    assert all_gone([int(p) for p in line[0].split()[1:]], timeout=5)
    _remove_lock_files_of_killed_creators(r.stdout)


_PARENT_KILLED = """
import os, sys, time
import numpy as np
from dejaq import Parallel

def big(x):
    return np.ones(500_000)

if __name__ == "__main__":
    stage = Parallel(n_workers=2)(big)(range(100))  # blocks on its full queue
    if sys.argv[1] == "running":
        next(iter(stage))
    # else: the workers are probably still starting when this process is killed
    print("queues", stage._out_queue._base, stage._it._out_queue._base, flush=True)
    print("workers", *(stage.pids + stage._it.pids), flush=True)
    time.sleep(60)
"""


@pytest.mark.parametrize("when", ["running", "starting"])
def test_workers_exit_when_the_main_process_is_killed(tmp_path, when):
    script = tmp_path / "script.py"
    script.write_text(_PARENT_KILLED)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    proc = subprocess.Popen([sys.executable, str(script), when], stdout=subprocess.PIPE, text=True, env=env)
    try:
        names = proc.stdout.readline()
        line = proc.stdout.readline()
        assert line.startswith("workers"), line
        pids = [int(p) for p in line.split()[1:]]
        proc.kill()  # e.g. a crash, or a Jupyter kernel that is killed
        proc.wait(10)
        assert all_gone(pids, timeout=10)
    finally:
        proc.kill()
        _remove_lock_files_of_killed_creators(names if "names" in locals() else "")


# ---------- small things ----------

def test_lambda_keyword_arguments():
    out = Parallel(n_workers=2)(lambda x, f: f(x))(range(4), f=lambda v: v * 10).run(progress=False)
    assert out == [0, 10, 20, 30]


_started = []


class _FailsAfterOneStart(parallel.ctx.Process):
    def start(self):
        if _started:
            raise OSError("too many open files")
        super().start()
        _started.append(self.pid)


def test_workers_are_stopped_if_starting_them_fails(monkeypatch):
    import gc
    upstream = Parallel(n_workers=1)(ident)(range(10))
    _started.clear()
    monkeypatch.setattr(parallel.ctx, "Process", _FailsAfterOneStart)
    with pytest.raises(OSError):
        lazymap(add5, upstream, n_workers=2)
    gc.collect()
    assert len(_started) == 1 and all_gone(_started)


# ---------- ordered=False ----------

def slow_every_4th(x):
    time.sleep(0.2 if x % 4 == 0 else 0.0)
    return x


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("n_workers", [2, 3, 4])
@pytest.mark.parametrize("n_items", [0, 1, 5, 12])
def test_item_counts(n_workers, n_items, ordered):
    out = Parallel(n_workers=n_workers, ordered=ordered)(add5)(range(n_items)).run(progress=False)
    assert (out if ordered else sorted(out)) == [i + 5 for i in range(n_items)]


@pytest.mark.parametrize("source", ["range", "list", "stage"])
def test_unordered_returns_every_result_once(source):
    data = {"range": range(30), "list": list(range(30)), "stage": Parallel(n_workers=2)(ident)(range(30))}[source]
    out = Parallel(n_workers=3, ordered=False)(add5)(data).run(progress=False)
    assert sorted(out) == [i + 5 for i in range(30)]


def test_unordered_slow_items_do_not_hold_up_the_others():
    out = Parallel(n_workers=4, ordered=False)(slow_every_4th)(list(range(8))).run(progress=False)
    assert sorted(out) == list(range(8))
    assert out.index(0) > out.index(1)  # item 0 is slow: later items overtake it


def test_unordered_errors_and_downstream_stages():
    with pytest.raises(ValueError, match="bad item 3"):
        Parallel(n_workers=3, ordered=False)(fails_at_3)(range(10)).run(progress=False)
    up = Parallel(n_workers=3, ordered=False)(add5)(range(20))
    assert sorted(Parallel(n_workers=2)(ident)(up).run(progress=False)) == [i + 5 for i in range(20)]


# ---------- Startable and Ticker as the input of a stage ----------

def test_startable_input_waits_for_start():
    from dejaq.parallel import Startable
    s = Startable(range(5))
    stage = Parallel(n_workers=1)(add5)(s)
    it = iter(stage)
    time.sleep(1.0)
    assert stage._out_queue.nitems == 0  # nothing before start()
    s.start()
    assert [next(it) for _ in range(5)] == [5, 6, 7, 8, 9]


def test_startable_emit_one_and_stop_across_processes():
    from dejaq.parallel import Startable
    s = Startable(range(100), emit_one=True)
    stage = Parallel(n_workers=2)(add5)(s)
    it = iter(stage)
    got = []
    for _ in range(3):
        s.start()
        got.append(next(it))
    time.sleep(0.5)
    assert got == [5, 6, 7] and stage._out_queue.nitems == 0  # one item per start()
    s.stop()
    assert list(it) == []


def test_startable_pause():
    from dejaq.parallel import Startable
    s = Startable(range(1000))
    it = iter(s)
    s.start()
    assert next(it) == 0
    s.pause()
    import threading
    out = []
    t = threading.Thread(target=lambda: out.append(next(it)), daemon=True)
    t.start()
    t.join(0.5)
    assert not out  # waits while paused
    s.start()
    t.join(5)
    assert out == [1]


def test_ticker_as_stage_input():
    import datetime
    from dejaq.parallel import Ticker
    t = Ticker(emit_one=True)
    stage = Parallel(n_workers=1)(ident)(t)
    it = iter(stage)
    t.start()
    assert isinstance(next(it), datetime.datetime)
    t.stop()
    assert list(it) == []


# ---------- zip and pass-through stages forward with one copy ----------

def test_zip_ends_with_the_shortest_input():
    a = Parallel(n_workers=2)(add5)(range(10))
    b = Parallel(n_workers=1)(ident)(range(4))
    assert parallel.zip(a, b).run(progress=False) == [(5, 0), (6, 1), (7, 2), (8, 3)]


def test_zip_input_error_is_raised():
    a = Parallel(n_workers=1)(ident)(range(10))
    b = Parallel(n_workers=2)(fails_at_3)(range(10))
    with pytest.raises(ValueError, match="bad item 3"):
        parallel.zip(a, b).run(progress=False)


def test_zip_of_tee_branches_with_frames():
    up = Parallel(n_workers=1, buffer_bytes=50e6)(big)(range(10))
    a, b = up.tee(2)
    out = Parallel(n_workers=2)(lambda ab: float(ab[0].sum() + ab[1].sum()))(parallel.zip(a, b)).run(progress=False)
    assert out == [1_000_000.0] * 10


def test_pass_through_stage():
    up = Parallel(n_workers=2)(add5)(range(10))
    assert lazymap(None, up).run(progress=False) == [i + 5 for i in range(10)]


# ---------- zero_copy ----------

def numbered_frame(i):
    return np.full(250_000, i, dtype=np.int64)  # 2 MB


def half_view(frame):
    return frame[::2]  # a view of the input: put before the input's memory is released


def frame_sum(frame):
    return int(frame[0]), int(frame.sum())


def test_zero_copy_stage_results_stay_correct_when_the_ring_wraps():
    cam = Parallel(n_workers=1, buffer_bytes=9e6)(numbered_frame)(range(60))  # ~4 frames per ring
    views = Parallel(n_workers=1, zero_copy=True)(half_view)(cam)
    sums = Parallel(n_workers=1, zero_copy=True)(frame_sum)(views)
    assert sums.run(progress=False) == [(i, i * 125_000) for i in range(60)]


def test_zero_copy_stage_errors_and_zip_input():
    a = Parallel(n_workers=1)(ident)(range(10))
    with pytest.raises(ValueError, match="bad item 3"):
        Parallel(n_workers=1, zero_copy=True)(fails_at_3)(a).run(progress=False)
    a, b = Parallel(n_workers=1)(ident)(range(5)), Parallel(n_workers=1)(add5)(range(5))
    assert Parallel(n_workers=1, zero_copy=True)(sum)(parallel.zip(a, b)).run(progress=False) == [5, 7, 9, 11, 13]


def test_zero_copy_needs_one_worker():
    with pytest.raises(ValueError, match="n_workers=1"):
        Parallel(n_workers=2, zero_copy=True)(ident)(range(3))


def test_zero_copy_with_a_plain_input_works_as_usual():
    assert Parallel(n_workers=1, zero_copy=True)(add5)(range(3)).run(progress=False) == [5, 6, 7]


def test_iterating_without_copies_in_the_main_process():
    stage = Parallel(n_workers=1, buffer_bytes=9e6)(numbered_frame)(range(40))
    seen = []
    for i, frame in enumerate(stage.iter(zero_copy=True)):
        seen.append((int(frame[0]), int(frame[-1])))  # valid until the next frame is requested
        if i == 9:
            break
    for frame in stage.iter(zero_copy=True):  # continues with the next frame
        seen.append((int(frame[0]), int(frame[-1])))
    assert seen == [(i, i) for i in range(40)]


def test_iterating_without_copies_raises_worker_errors():
    with pytest.raises(ValueError, match="bad item 3"):
        for _ in Parallel(n_workers=2)(fails_at_3)(range(10)).iter(zero_copy=True):
            pass


_CLOSE_STRESS = """
import numpy as np
from dejaq import Parallel, parallel

def big(x):
    return np.ones(500_000)

def ident(x):
    return x

if __name__ == "__main__":
    for i in range(12):
        a = Parallel(n_workers=2)(big)(range(1000))
        b = Parallel(n_workers=1)(big)(range(1000))
        kind = i % 3
        if kind == 0:
            last = Parallel(n_workers=2)(ident)(a)
        elif kind == 1:
            last = parallel.zip(a, b)
        else:
            x, y = a.tee(2)
            last = Parallel(n_workers=1)(ident)(x)
        next(iter(last))  # every process is busy moving data when they are stopped
        last.close()
        if kind == 2:
            y.close()
        b.close()
    print("ok", flush=True)
"""


def test_closing_busy_pipelines_never_hangs(tmp_path):
    r = run_script(tmp_path, _CLOSE_STRESS, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-3000:]
    assert "Traceback" not in r.stderr, r.stderr[-3000:]  # workers end quietly when their stage is closed


TEE_AT_EXIT = """
from dejaq import Parallel

def inc(x):
    return x + 1

if __name__ == "__main__":
    a, b = Parallel(n_workers=1)(inc)(range(5)).tee(2)
    assert list(zip(a, b)) == [(i, i) for i in range(1, 6)]
"""


def test_tee_left_open_at_exit_prints_no_error(tmp_path):
    r = run_script(tmp_path, TEE_AT_EXIT)
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stderr, r.stderr
