"""Importing dejaq must not change global state of the importing program (start method, logging)."""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_python(code):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env)


def test_import_has_no_global_side_effects():
    code = (
        "import logging, multiprocessing as mp\n"
        "import dejaq, dejaq.parallel, dejaq.remote, dejaq.stream\n"
        "root = logging.getLogger()\n"
        "print(mp.get_start_method(allow_none=True), len(root.handlers), root.level)\n"
    )
    r = _run_python(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["None", "0", str(30)]  # start method unset, root logger untouched (WARNING)


def test_errors_are_visible_without_logging_config():
    """With no logging configured by the application, dejaq's errors still reach stderr (and INFO does not)."""
    code = (
        "import logging, dejaq.remote\n"
        "logging.getLogger('dejaq.remote').error('visible error')\n"
        "logging.getLogger('dejaq.remote').info('hidden info')\n"
    )
    r = _run_python(code)
    assert r.returncode == 0 and "visible error" in r.stderr and "hidden info" not in r.stderr + r.stdout
