"""Detect OS objects that a test run leaves behind.

    python tests/check_leftovers.py snapshot <file>
    python -m pytest ...
    python tests/check_leftovers.py compare <file>   # lists new objects; exit code 1 if there are any

Linux: everything in /dev/shm (POSIX shared memory, semaphores as sem.*, dejaq's lock files).
macOS: dejaq's lock files in /tmp (POSIX shared memory and semaphores cannot be listed there).
Windows needs no such check: kernel objects disappear with the last process holding them.
"""

import glob
import os
import sys
import time


def _objects() -> set:
    if os.path.isdir("/dev/shm"):
        # multiprocessing unlinks its own semaphores right after creating them; skip any in flight
        return {name for name in os.listdir("/dev/shm") if not name.startswith("sem.mp-")}
    return {os.path.basename(p) for p in glob.glob("/tmp/dejaq.*.lock")}


def main(argv) -> int:
    if len(argv) != 3 or argv[1] not in ("snapshot", "compare"):
        print(__doc__)
        return 2
    path = argv[2]
    if argv[1] == "snapshot":
        with open(path, "w") as f:
            f.write("\n".join(sorted(_objects())))
        return 0
    with open(path) as f:
        before = set(f.read().split())
    deadline = time.monotonic() + 15  # resource trackers of ended processes clean up asynchronously
    while True:
        new = sorted(_objects() - before)
        if not new or time.monotonic() > deadline:
            break
        time.sleep(0.5)
    if new:
        print(f"{len(new)} OS object(s) left behind by the test run:")
        for name in new:
            print("  " + name)
        return 1
    print("No OS objects left behind.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
