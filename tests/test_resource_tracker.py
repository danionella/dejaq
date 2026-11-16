"""Attaching to a queue must not hand its shared memory to the attacher's resource tracker (POSIX).

On POSIX, ``SharedMemory(name=...)`` registers the segment with the calling process's
resource_tracker. A process started independently (not a multiprocessing child) has its own
tracker, which unlinks every registered segment when that process exits. If attachers register,
the first independent consumer to exit destroys the creator's queue for everyone else.
"""

import os
import pickle
import subprocess
import sys
import textwrap

import pytest

from dejaq.queues import DejaQueue

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX resource_tracker only")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [_REPO_ROOT, os.environ.get("PYTHONPATH")])))

_GET_ONE = """
import pickle, sys
q = pickle.loads(sys.stdin.buffer.read())
print(q.get(timeout=20))
"""


def _run(script, stdin=b""):
    proc = subprocess.run(
        [sys.executable, "-c", script], input=stdin, capture_output=True, env=_ENV, timeout=60
    )
    return proc.returncode, proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace")


def test_queue_survives_exit_of_independent_attacher():
    q = DejaQueue(buffer_bytes=1_000_000)
    q.put("first")
    q.put("second")
    blob = pickle.dumps(q)

    for expected in ("first", "second"):
        # Each child attaches by pickle in a fresh interpreter (own resource tracker), takes one
        # item and exits. communicate() waits for the child's tracker too, since it shares stderr.
        rc, out, err = _run(_GET_ONE, blob)
        assert rc == 0, err
        assert out.strip() == expected
        assert "leaked" not in err, err

    # the creator can still use the queue
    q.put("third")
    assert q.get(timeout=5) == "third"


_MP_SCRIPT = textwrap.dedent(
    """
    import gc, multiprocessing as mp, sys
    from dejaq.queues import DejaQueue, NamedByteRing

    def child(q):
        assert q.get(timeout=20) == "hello"

    if __name__ == "__main__":
        ctx = mp.get_context(sys.argv[1])
        q = DejaQueue(buffer_bytes=1_000_000)
        q.put("hello")
        p = ctx.Process(target=child, args=(q,))
        p.start()
        p.join(30)
        assert p.exitcode == 0, p.exitcode
        base = q._base
        del q
        gc.collect()
        # the creator unlinked its segments on deletion
        try:
            NamedByteRing(name=base, create=False)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("shared memory still exists after the creator was deleted")
        print("ok")
    """
)


@pytest.mark.parametrize("method", ["spawn", "forkserver"])  # both hand the queue over by pickle
def test_multiprocessing_child_shares_creator_tracker(method, tmp_path):
    # A multiprocessing child shares its parent's resource tracker. Attaching in the child must
    # not remove the creator's registration, or the creator's unlink() makes the tracker raise
    # (KeyError traceback on stderr) and crash cleanup is lost.
    script = tmp_path / "mp_script.py"
    script.write_text(_MP_SCRIPT)
    proc = subprocess.run([sys.executable, str(script), method], capture_output=True, env=_ENV, timeout=90)
    out, err = proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace")
    assert proc.returncode == 0, err
    assert out.strip() == "ok"
    for bad in ("Traceback", "KeyError", "leaked"):
        assert bad not in err, err
