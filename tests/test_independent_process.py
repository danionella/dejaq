"""DejaQueue shared with a separately started Python process (not a multiprocessing child).

The queue is handed over only as its pickle (i.e. the names of its shared memory and named
semaphores), which the other process uses to attach. Set DEJAQ_OTHER_PYTHON to the path of a
different interpreter (e.g. a free-threaded build) to also run the child with that interpreter.
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
while True:
    item = q_in.get(timeout=20)
    if item is Ellipsis:
        break
    q_out.put({"echo": item, "sum": float(np.sum(item["data"])), "gil": getattr(sys, "_is_gil_enabled", lambda: True)()})
q_out.put(Ellipsis)
"""

_PYTHONS = [pytest.param(sys.executable, id="same-python")]
if os.environ.get("DEJAQ_OTHER_PYTHON"):
    _PYTHONS.append(pytest.param(os.environ["DEJAQ_OTHER_PYTHON"], id="other-python"))


@pytest.mark.parametrize("python", _PYTHONS)
def test_dejaqueue_with_independent_process(python):
    q_in = DejaQueue(buffer_bytes=4_000_000)
    q_out = DejaQueue(buffer_bytes=4_000_000)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [_REPO_ROOT, os.environ.get("PYTHONPATH")])))
    proc = subprocess.Popen(
        [python, "-c", _CHILD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )
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
                _, err = proc.communicate()
                pytest.fail("no reply from child process:\n" + err.decode(errors="replace"))
            if item is Ellipsis:
                break
            received.append(item)

        _, err = proc.communicate(timeout=30)
        assert proc.returncode == 0, err.decode(errors="replace")
    finally:
        if proc.poll() is None:
            proc.kill()

    assert len(received) == len(sent)
    for s, r in zip(sent, received):
        assert r["echo"]["i"] == s["i"]
        np.testing.assert_array_equal(r["echo"]["data"], s["data"])
        assert r["sum"] == pytest.approx(float(np.sum(s["data"])))
