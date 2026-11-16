"""DejaQueue shared with a separately started Python process (not a multiprocessing child).

The queue is handed over only as its pickle (i.e. the names of its shared memory and named
semaphores), which the other process uses to attach. Set DEJAQ_OTHER_PYTHON to the path of a
different interpreter build (e.g. free-threaded vs. regular) to also run the child with that one.
"""

import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from dejaq.queues import DejaQueue

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHILD = r"""
import pickle, sys
import numpy as np
q_in, q_out = pickle.loads(sys.stdin.buffer.read())
q_out.put((sys.version, getattr(sys, "_is_gil_enabled", lambda: True)()))  # same as _build() below
while True:
    item = q_in.get(timeout=20)
    if item is Ellipsis:
        break
    q_out.put({"echo": item, "sum": float(np.sum(item["data"]))})
q_out.put(Ellipsis)
"""


def _build():
    return (sys.version, getattr(sys, "_is_gil_enabled", lambda: True)())


_PYTHONS = [pytest.param(sys.executable, id="same-python")]
if os.environ.get("DEJAQ_OTHER_PYTHON"):
    _PYTHONS.append(pytest.param(os.environ["DEJAQ_OTHER_PYTHON"], id="other-python"))


@pytest.mark.parametrize("python", _PYTHONS)
def test_dejaqueue_with_independent_process(python, tmp_path):
    q_in = DejaQueue(buffer_bytes=4_000_000)
    q_out = DejaQueue(buffer_bytes=4_000_000)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [_REPO_ROOT, os.environ.get("PYTHONPATH")])))
    err_path = tmp_path / "child_stderr.txt"
    with open(err_path, "wb") as err_file:  # a file, not a pipe: a chatty child can't block on a full pipe
        proc = subprocess.Popen(
            [python, "-c", _CHILD],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=err_file,
            env=env,
        )

    def child_stderr():
        return err_path.read_text(errors="replace")

    try:
        proc.stdin.write(pickle.dumps((q_in, q_out)))
        proc.stdin.close()

        rng = np.random.default_rng(0)
        sent = [{"i": i, "data": rng.standard_normal((64, 64))} for i in range(20)]
        for item in sent:
            q_in.put(item)
        q_in.put(Ellipsis)

        received = []
        while True:
            try:
                item = q_out.get(timeout=30)
            except TimeoutError:
                proc.kill()
                proc.wait()
                pytest.fail("no reply from child process:\n" + child_stderr())
            if item is Ellipsis:
                break
            received.append(item)

        assert proc.wait(timeout=30) == 0, child_stderr()
    finally:
        if proc.poll() is None:
            proc.kill()

    child_build, received = received[0], received[1:]
    if python != sys.executable:  # make sure the "other" interpreter really is a different build
        assert child_build != _build(), f"DEJAQ_OTHER_PYTHON runs the same build as the parent: {child_build}"
    assert len(received) == len(sent)
    for s, r in zip(sent, received):
        assert r["echo"]["i"] == s["i"]
        np.testing.assert_array_equal(r["echo"]["data"], s["data"])
        assert r["sum"] == pytest.approx(float(np.sum(s["data"])))
