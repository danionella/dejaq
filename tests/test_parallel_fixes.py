"""Regression tests for the dejaq.parallel fixes (numbered as in the review).

Anything that used to hang runs in a separate interpreter with a time limit.
"""

import gc
import os
import re
import subprocess
import sys
import textwrap
import time

import numpy as np
import psutil
import pytest

from dejaq import Parallel, parallel
from dejaq.parallel import WorkerError, lazymap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_script(tmp_path, code, timeout=60):
    script = tmp_path / "script.py"
    script.write_text(code)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    try:
        return subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the program did not finish within {timeout} s (hang)")


def _all_gone(pids, timeout=10.0):
    """True once none of the processes is running any more (zombies count as gone)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = []
        for pid in pids:
            try:
                if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                    alive.append(pid)
            except psutil.NoSuchProcess:
                pass
        if not alive:
            return True
        time.sleep(0.1)
    return False


def _stage_pids(stage):
    return stage.pids + (stage._it.pids if isinstance(stage._it, lazymap) else [])


def add5(x):
    return x + 5


def big(x):
    return np.ones(500_000)  # 4 MB: the 10 MB output queue holds 2


def fails_at_3(x):
    if x == 3:
        raise ValueError("bad item 3")
    return x


# ---------- 1: run() as documented ----------

def test_1_run_returns_a_list():
    assert Parallel(n_workers=2)(add5)(range(5)).run(progress=False) == [5, 6, 7, 8, 9]


def _readme_parallel_code():
    readme = open(os.path.join(REPO_ROOT, "README.md")).read()
    section = readme.split("## dejaq.Parallel", 1)[1].split("\n# ", 1)[0]
    return "\n".join(re.findall(r"```python\n(.*?)```", section, flags=re.S))


def test_1_readme_examples_run(tmp_path):
    code = (
        "import time\n"
        "time.sleep = lambda seconds: None  # the examples simulate slow work\n"
        "if __name__ == '__main__':\n"
        + textwrap.indent(_readme_parallel_code(), "    ")
        + "\n    print('ok')\n"
    )
    r = _run_script(tmp_path, code, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


# ---------- 2: leaving early, garbage collection, exit ----------

def test_2_leaving_a_loop_early_and_iterating_again_continues():
    stage = Parallel(n_workers=2)(add5)(range(10))
    for x in stage:
        break
    assert x == 5
    assert list(stage) == [6, 7, 8, 9, 10, 11, 12, 13, 14]
    stage.close()
    assert _all_gone(_stage_pids(stage))


def test_2_iterating_a_closed_stage_raises():
    stage = Parallel(n_workers=2)(big)(range(100))
    next(iter(stage))
    stage.close()
    with pytest.raises(RuntimeError, match="closed"):
        list(stage)


def test_2_close_wakes_a_waiting_consumer():
    """A consumer waiting for results in another thread (here: submit()) raises instead of waiting forever."""
    stage = Parallel(n_workers=1)(time.sleep)([30])
    future = stage.submit()
    time.sleep(0.5)
    stage.close()
    with pytest.raises(RuntimeError, match="closed"):
        future.result(timeout=10)


def test_2_garbage_collected_stage_stops_its_workers():
    stage = Parallel(n_workers=2)(big)(range(100))
    next(iter(stage))
    pids = _stage_pids(stage)
    del stage
    gc.collect()
    assert _all_gone(pids)


_EARLY_EXIT = """
import numpy as np
from dejaq import Parallel

def big(i):
    return np.ones(500_000)

if __name__ == "__main__":
    for i, x in enumerate(Parallel(n_workers=2)(big)(range(100))):
        if i == 2:
            break
    stage = Parallel(n_workers=2)(big)(range(100))
    for i, x in enumerate(stage):  # left early, still referenced at exit
        if i == 2:
            break
    stage = Parallel(n_workers=2)(big)(range(100))  # not iterated to the end, still referenced at exit
    next(iter(stage))
    print("exiting", flush=True)
"""


def test_2_program_exits_after_leaving_early(tmp_path):
    """Formerly the interpreter waited forever at exit for workers blocked on a full queue."""
    r = _run_script(tmp_path, _EARLY_EXIT)
    assert r.returncode == 0 and "exiting" in r.stdout, r.stderr[-3000:]


# ---------- 3: a worker that dies ----------

_DYING_WORKER = """
import os, signal, sys
from dejaq import Parallel
from dejaq.parallel import WorkerError

def dies_at_3(x):
    if x == 3:
        if sys.argv[1] == "kill" and hasattr(signal, "SIGKILL"):
            os.kill(os.getpid(), signal.SIGKILL)  # e.g. the out-of-memory killer
        if sys.argv[1] == "exit0":
            os._exit(0)  # exit code 0, but the worker did not finish
        os._exit(3)  # any abnormal exit
    return x

def add1(x):
    return x + 1

if __name__ == "__main__":
    if sys.argv[2] == "single":
        stage = Parallel(n_workers=2)(dies_at_3)(range(10))
    else:  # the worker dies in an upstream stage
        stage = Parallel(n_workers=2)(add1)(Parallel(n_workers=2)(dies_at_3)(range(10)))
    try:
        stage.run(progress=False)
    except WorkerError as e:
        print("WorkerError:", e, flush=True)
"""


@pytest.mark.parametrize("how", ["exit", "kill", "exit0"])
@pytest.mark.parametrize("where", ["single", "upstream"])
def test_3_dead_worker_raises_instead_of_hanging(tmp_path, how, where):
    script = tmp_path / "script.py"
    script.write_text(_DYING_WORKER)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    try:
        r = subprocess.run([sys.executable, str(script), how, where], capture_output=True, text=True,
                           timeout=60, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("hung instead of raising WorkerError")
    assert "WorkerError:" in r.stdout and "died" in r.stdout, (r.stdout, r.stderr[-3000:])
    assert r.returncode == 0, r.stderr[-3000:]


# ---------- 4: exceptions ----------

def test_4_worker_exception_keeps_its_type_and_traceback():
    stage = Parallel(n_workers=2)(fails_at_3)(range(10))
    with pytest.raises(ValueError, match="bad item 3") as info:
        stage.run(progress=False)
    assert "fails_at_3" in str(info.value.__cause__)  # the worker's traceback
    assert _all_gone(_stage_pids(stage))  # the stage was stopped


def test_4_exception_from_an_upstream_stage_reaches_the_end():
    stage = Parallel(n_workers=2)(add5)(Parallel(n_workers=2)(fails_at_3)(range(10)))
    with pytest.raises(ValueError, match="bad item 3"):
        stage.run(progress=False)


class _Unpicklable(Exception):
    def __reduce__(self):
        raise TypeError("cannot pickle this exception")


def raises_unpicklable(x):
    raise _Unpicklable("special")


def test_4_unpicklable_exception_becomes_worker_error():
    with pytest.raises(WorkerError, match="_Unpicklable: special"):
        Parallel(n_workers=1)(raises_unpicklable)(range(3)).run(progress=False)


def calls_sys_exit(x):
    if x == 3:
        sys.exit()
    return x


def test_4_sys_exit_in_a_worker_is_raised():
    with pytest.raises(SystemExit):
        Parallel(n_workers=2)(calls_sys_exit)(range(10)).run(progress=False)


class _FailsToIterate:
    def __iter__(self):
        raise ValueError("cannot iterate")


def test_4_error_while_a_worker_starts_keeps_its_type():
    with pytest.raises(ValueError, match="cannot iterate"):
        lazymap(add5, _FailsToIterate(), n_workers=1).run(progress=False)


def looks_like_an_error(x):
    return {"type": "exception", "exc_type": "fake", "exc_msg": "", "traceback": "", "i": x}


def test_4_dict_results_are_never_mistaken_for_errors():
    out = Parallel(n_workers=2)(looks_like_an_error)(range(3)).run(progress=False)
    assert [d["i"] for d in out] == [0, 1, 2]


# ---------- 5: close(), zip, compute() ----------

def test_5_close_and_context_manager():
    stage = Parallel(n_workers=2)(big)(range(100))
    next(iter(stage))
    stage.close()
    stage.close()  # again: no error
    assert _all_gone(_stage_pids(stage))
    with Parallel(n_workers=2)(add5)(range(3)) as stage2:
        assert stage2.run(progress=False) == [5, 6, 7]
    assert _all_gone(_stage_pids(stage2))


def pair_sum(pair):
    return pair[0] + pair[1]


def test_5_zip():
    a = Parallel(n_workers=2)(add5)(range(5))
    b = Parallel(n_workers=2)(add5)(range(10, 15))
    assert parallel.zip(a, b).run(progress=False) == [(5, 15), (6, 16), (7, 17), (8, 18), (9, 19)]
    a = Parallel(n_workers=2)(add5)(range(5))
    b = Parallel(n_workers=2)(add5)(range(10, 15))
    assert Parallel(n_workers=2)(pair_sum)(parallel.zip(a, b)).run(progress=False) == [20, 22, 24, 26, 28]


def make_dict(x):
    return {"x": x}


def test_5_compute_returns_a_list_by_default():
    out = Parallel(n_workers=2)(make_dict)(range(3)).compute(progress=False)
    assert out == [{"x": 0}, {"x": 1}, {"x": 2}]
    arr = lazymap(add5, range(3), n_workers=2).compute(progress=False, ndarray=True)
    assert isinstance(arr, np.ndarray) and arr.tolist() == [5, 6, 7]


# ---------- found in the review of the fixes ----------

def jittered(x):
    time.sleep(0.003 * ((x * 7919) % 5))
    return x


def test_chained_multi_worker_stages_keep_order_and_items():
    """Each downstream worker used to start the upstream stage, so several input tokens circulated."""
    for _ in range(5):
        stage = Parallel(n_workers=3)(jittered)(Parallel(n_workers=3)(jittered)(range(60)))
        assert stage.run(progress=False) == list(range(60))


def to_ellipsis(x):
    return ... if x == 1 else x


def test_ellipsis_items_and_results_pass_through():
    assert lazymap(None, [0, ..., 2], n_workers=2).run(progress=False) == [0, ..., 2]
    assert Parallel(n_workers=2)(to_ellipsis)(range(4)).run(progress=False) == [0, ..., 2, 3]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        lazymap(add5, range(3), n_workers=0)
    with pytest.raises(ValueError):
        lazymap(add5, range(3), start_type="Lazy")
