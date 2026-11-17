"""Tests for tee(), closing a pipeline, and worker processes that are left behind."""

import os
import subprocess
import sys
import time

import numpy as np
import psutil
import pytest

from dejaq import Parallel, parallel
from dejaq.parallel import lazymap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_script(tmp_path, code, timeout=60):
    script = tmp_path / "script.py"
    script.write_text(code)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    try:
        return subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the program did not finish within {timeout} s (hang)")


def _alive(pids):
    n = 0
    for pid in pids:
        try:
            if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                n += 1
        except psutil.NoSuchProcess:
            pass
    return n


def _all_gone(pids, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pids):
            return True
        time.sleep(0.1)
    return False


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
    assert _all_gone(up.pids + up._own_input.pids + a.pids)


def test_tee_garbage_collected_branches_stop_the_tee_and_the_stage():
    import gc
    up = Parallel(n_workers=2)(big)(range(20))
    pids = up.pids + up._own_input.pids
    a, b = up.tee(2)
    pids += a.pids
    next(iter(a))
    del up, a, b
    gc.collect()
    assert _all_gone(pids)


# ---------- closing a pipeline ----------

def test_closing_the_last_stage_closes_the_stages_that_feed_it():
    stage1 = Parallel(n_workers=2)(big)(range(100))
    stage2 = Parallel(n_workers=2)(ident)(stage1)
    stage3 = Parallel(n_workers=1)(ident)(stage2)
    next(iter(stage3))
    stage3.close()
    assert _all_gone(stage1.pids + stage1._own_input.pids + stage2.pids + stage3.pids)
    with pytest.raises(RuntimeError, match="closed"):
        list(stage1)


def test_an_error_closes_the_whole_pipeline():
    stage1 = Parallel(n_workers=2)(fails_at_3)(range(100))
    stage2 = Parallel(n_workers=2)(ident)(stage1)
    with pytest.raises(ValueError):
        stage2.run(progress=False)
    assert _all_gone(stage1.pids + stage1._own_input.pids + stage2.pids)


def test_closing_a_zip_closes_its_inputs():
    a = Parallel(n_workers=2)(big)(range(100))
    b = Parallel(n_workers=1)(big)(range(100))
    z = parallel.zip(a, b)
    next(iter(z))
    z.close()
    assert _all_gone(a.pids + a._own_input.pids + b.pids + z.pids)


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
    r = _run_script(tmp_path, code)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("grandchildren")]
    assert line, (r.stdout, r.stderr[-3000:])
    assert _all_gone([int(p) for p in line[0].split()[1:]], timeout=5)


_PARENT_KILLED = """
import os, sys, time
import numpy as np
from dejaq import Parallel

def big(x):
    return np.ones(500_000)

if __name__ == "__main__":
    stage = Parallel(n_workers=2)(big)(range(100))  # blocks on its full queue
    next(iter(stage))
    print("workers", *(stage.pids + stage._own_input.pids), flush=True)
    time.sleep(60)
"""


def test_workers_exit_when_the_main_process_is_killed(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(_PARENT_KILLED)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True, env=env)
    try:
        line = proc.stdout.readline()
        assert line.startswith("workers"), line
        pids = [int(p) for p in line.split()[1:]]
        proc.kill()  # e.g. a crash, or a Jupyter kernel that is killed
        proc.wait(10)
        assert _all_gone(pids, timeout=10)
    finally:
        proc.kill()


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
    assert len(_started) == 1 and _all_gone(_started)


# ---------- range inputs read by stride, ordered=False ----------

def slow_every_4th(x):
    time.sleep(0.2 if x % 4 == 0 else 0.0)
    return x


@pytest.mark.parametrize("n_workers", [2, 3, 4])
@pytest.mark.parametrize("n_items", [0, 1, 5, 12])
def test_range_input_split_between_workers(n_workers, n_items):
    stage = Parallel(n_workers=n_workers)(add5)(range(n_items))
    assert stage._own_input is None  # each worker reads its share: no extra input stage
    assert stage.run(progress=False) == [i + 5 for i in range(n_items)]


def test_range_input_with_a_step():
    assert Parallel(n_workers=3)(ident)(range(3, 40, 4)).run(progress=False) == list(range(3, 40, 4))


@pytest.mark.parametrize("source", ["range", "list", "stage"])
def test_unordered_returns_every_result_once(source):
    data = {"range": range(30), "list": list(range(30)), "stage": Parallel(n_workers=2)(ident)(range(30))}[source]
    out = Parallel(n_workers=3, ordered=False)(add5)(data).run(progress=False)
    assert sorted(out) == [i + 5 for i in range(30)]


def test_unordered_slow_items_do_not_hold_up_the_others():
    out = Parallel(n_workers=4, ordered=False)(slow_every_4th)(range(8)).run(progress=False)
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
