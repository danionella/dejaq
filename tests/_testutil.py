"""Helpers shared by the tests."""

import os
import subprocess
import sys
import time

import psutil
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(tmp_path, code, *args, timeout=60):
    """Run code as a script file (needed for the spawn start method) in a fresh interpreter; fail on a hang."""
    script = tmp_path / "script.py"
    script.write_text(code)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    try:
        return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the program did not finish within {timeout} s (hang)")


def all_gone(pids, timeout=10.0):
    """True once none of the processes is running any more (zombies count as gone)."""
    def running(pid):
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    deadline = time.monotonic() + timeout
    while any(running(p) for p in pids):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True
