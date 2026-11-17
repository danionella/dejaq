"""A/B benchmark of DejaQueue and dejaq.parallel: compares two checkouts of dejaq on the same machine.

    python benchmarks/ab_bench.py --rounds 5 base=<path to checkout> head=<path to checkout> [--markdown out.md]
        [--suite queue|pipeline]

Each round runs every case once per variant, each variant in a fresh process with that checkout on
PYTHONPATH; the order of the variants alternates between rounds, so slow drifts of a (shared) machine
affect both alike. The report gives medians and per-round paired ratios (head / base, for times).
Only the public API is used, so any dejaq version can be compared.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

# ----------------------------------------------------------------------------------------------- cases


def _payload(kind):
    import numpy as np

    return {
        "int": lambda: 7,
        "small dict": lambda: {"a": 1, "b": [1, 2, 3], "s": "text", "f": 1.5},
        "dict 10k entries": lambda: {f"k{i}": i for i in range(10_000)},
        "200 dicts w/ arrays": lambda: [{"i": i, "x": np.ones(100)} for i in range(200)],
        "array 1MB": lambda: np.ones(2**20 // 8),
        "array 50MB": lambda: np.ones(50 * 2**20 // 8),
    }[kind]()


def _producer(q, kind, n, start):
    item = _payload(kind)
    start.wait()
    for _ in range(n):
        q.put(item)


def _consumer(q, n, done):
    for _ in range(n):
        q.get()
    done.put(time.perf_counter())


def _xproc(kind, n, nprod, ncons, buf):
    """Throughput (items/s) of nprod producer and ncons consumer processes."""
    import multiprocessing as mp
    from dejaq import DejaQueue

    ctx = mp.get_context("spawn")
    q = DejaQueue(buf)
    start, done = ctx.Event(), ctx.Queue()
    ps = [ctx.Process(target=_producer, args=(q, kind, n // nprod, start)) for _ in range(nprod)]
    cs = [ctx.Process(target=_consumer, args=(q, n // ncons, done)) for _ in range(ncons)]
    for p in ps + cs:
        p.start()
    time.sleep(1.0)  # let the workers start (slow on Windows and macOS)
    t0 = time.perf_counter()
    start.set()
    t1 = max(done.get() for _ in cs)
    for p in ps + cs:
        p.join()
    return n / (t1 - t0)


def _inproc(kind, n):
    """Time (us) of one put + get in a single process."""
    from dejaq import DejaQueue

    q, item = DejaQueue(100e6), _payload(kind)
    for _ in range(n // 10):
        q.put(item)
        q.get()
    t0 = time.perf_counter()
    for _ in range(n):
        q.put(item)
        q.get()
    return (time.perf_counter() - t0) / n * 1e6


def _get_cost(kind, n):
    """Time (us) of one get from a prefilled queue."""
    import pickle
    from dejaq import DejaQueue

    item = _payload(kind)
    q = DejaQueue((len(pickle.dumps(item, protocol=5)) + 1000) * n * 1.1 + 1e6)
    for _ in range(n):
        q.put(item)
    t0 = time.perf_counter()
    for _ in range(n):
        q.get()
    return (time.perf_counter() - t0) / n * 1e6


def _lifecycle(n=100):
    """Time (ms) to create, use and drop a queue in a program with many live objects."""
    from dejaq import DejaQueue

    junk = [[i] * 10 for i in range(200_000)]  # many live objects make full garbage collections expensive
    t0 = time.perf_counter()
    for _ in range(n):
        q = DejaQueue(65536)
        q.put(1)
        q.get()
        del q
    t = (time.perf_counter() - t0) / n * 1e3
    del junk
    return t


# (name, unit, higher is better, function)
CASES = [
    ("put+get int (1 process)", "us", False, lambda: _inproc("int", 30_000)),
    ("put+get small dict (1 process)", "us", False, lambda: _inproc("small dict", 30_000)),
    ("put+get 1MB array (1 process)", "us", False, lambda: _inproc("array 1MB", 2_000)),
    ("get dict 10k entries", "us", False, lambda: _get_cost("dict 10k entries", 200)),
    ("get 200 dicts w/ arrays", "us", False, lambda: _get_cost("200 dicts w/ arrays", 200)),
    ("1P1C int", "items/s", True, lambda: _xproc("int", 20_000, 1, 1, 50e6)),
    ("2P2C int", "items/s", True, lambda: _xproc("int", 20_000, 2, 2, 50e6)),
    ("1P1C 1MB", "GB/s", True, lambda: _xproc("array 1MB", 1_000, 1, 1, 200e6) * 2**20 / 1e9),
    ("2P2C 1MB small ring", "GB/s", True, lambda: _xproc("array 1MB", 1_000, 2, 2, 5e6) * 2**20 / 1e9),
    ("1P1C 50MB", "GB/s", True, lambda: _xproc("array 50MB", 40, 1, 1, 200e6) * 50 * 2**20 / 1e9),
    ("create+use+drop queue", "ms", False, _lifecycle),
]

# ------------------------------------------------------------------------------------------ pipelines
# Throughput of dejaq.parallel pipelines, measured once the workers have started (spawning them is not timed).


def _ident(x):
    return x


def _mb(x):
    import numpy as np

    return np.ones(2**20 // 8)


def _uneven(x):
    time.sleep(0.004 if x % 4 == 0 else 0.0005)  # every 4th task is slow: in-order output waits for it
    return x


def _consume(it):
    n = 0
    for _ in it:
        n += 1
    return n


def _timed(make, n, warmup=2.0):
    stages = make()
    time.sleep(warmup)  # let the workers start (they wait for the first item until iteration starts)
    t0 = time.perf_counter()
    got = _consume(stages[-1])
    t = time.perf_counter() - t0
    assert got == n, (got, n)
    for s in stages[::-1]:
        if hasattr(s, "close"):
            s.close()
    return n / t


def _stage(fcn, it, n_workers, buffer_bytes=10e6, **opts):
    from dejaq import Parallel

    return Parallel(n_workers=n_workers, buffer_bytes=buffer_bytes, **opts)(fcn)(it)


def _pipe_1(n_workers, n=20_000, as_list=False):
    return _timed(lambda: [_stage(_ident, list(range(n)) if as_list else range(n), n_workers)], n)


def _pipe_chain(n=20_000):
    def make():
        a = _stage(_ident, range(n), 1)
        return [a, _stage(_ident, a, 1)]
    return _timed(make, n)


def _pipe_mb(n_workers, n=2_000):
    return _timed(lambda: [_stage(_mb, range(n), n_workers)], n) * 2**20 / 1e9


def _pipe_1_4_1_mb(n=2_000):
    def make():
        a = _stage(_mb, range(n), 1)
        b = _stage(_ident, a, 4)
        return [a, b, _stage(_ident, b, 1)]
    return _timed(make, n) * 2**20 / 1e9


def _pipe_tee(n=20_000):
    def make():
        up = _stage(_ident, range(n), 1)  # referenced: dejaq <= 0.8 lost a temporary stage
        a, b = up.tee(2)
        return [up, a, b, zip(a, b)]
    return _timed(make, n)


def _pipe_zip(n=20_000):
    from dejaq import parallel

    def make():
        a, b = _stage(_ident, range(n), 1), _stage(_ident, range(n), 1)
        return [a, b, parallel.zip(a, b)]
    return _timed(make, n)


def _pipe_uneven(n=2_000, **opts):
    return _timed(lambda: [_stage(_uneven, range(n), 4, **opts)], n)


# camera-like: 4 MB frames from a driver buffer, queues of 100 MB (~25 frames), ending in sink stages that
# return a small value (as a writer would), so that the main process does not limit the throughput
_FRAME = None
_FRAME_BUF = 100e6


def _camera(i):
    global _FRAME
    if _FRAME is None:
        import numpy as np

        _FRAME = np.ones(4 * 2**20, np.uint8)
    return _FRAME


def _sink(x):
    return sum(int(f[-1]) for f in x) if isinstance(x, tuple) else int(x[-1])


def _fstage(fcn, it, n=1, zero_copy=False):
    from dejaq import Parallel

    opts = {"zero_copy": True} if zero_copy else {}  # (older versions have no zero_copy)
    return Parallel(n_workers=n, buffer_bytes=_FRAME_BUF, **opts)(fcn)(it)


def _frames(topology, zero_copy=False, n=1500):
    from dejaq import parallel

    def make():
        if topology == "zip":
            a, b = _fstage(_camera, range(n)), _fstage(_camera, range(n))
            z = parallel.zip(a, b)
            return [a, b, z, _fstage(_sink, z, zero_copy=zero_copy)]
        cam = _fstage(_camera, range(n))
        if topology == "tee":
            x, y = cam.tee(2)
            sx, sy = _fstage(_sink, x, zero_copy=zero_copy), _fstage(_sink, y, zero_copy=zero_copy)
            return [cam, x, y, sx, sy, zip(sx, sy)]
        if topology == "4 workers":
            mid = _fstage(_ident, cam, 4)
            return [cam, mid, _fstage(_sink, mid, zero_copy=zero_copy)]
        return [cam, _fstage(_sink, cam, zero_copy=zero_copy)]

    return _timed(make, n)


PIPELINE_CASES = [
    ("1 stage, 1 worker, int", "items/s", True, lambda: _pipe_1(1)),
    ("1 stage, 4 workers, int (range input)", "items/s", True, lambda: _pipe_1(4)),
    ("1 stage, 4 workers, int (list input)", "items/s", True, lambda: _pipe_1(4, as_list=True)),
    ("2 stages 1->1, int", "items/s", True, _pipe_chain),
    ("1 stage, 1 worker, 1MB results", "GB/s", True, lambda: _pipe_mb(1)),
    ("1 stage, 4 workers, 1MB results", "GB/s", True, lambda: _pipe_mb(4)),
    ("3 stages 1->4->1, 1MB", "GB/s", True, _pipe_1_4_1_mb),
    ("tee(2), int", "items/s", True, _pipe_tee),
    ("parallel.zip of 2 stages, int", "items/s", True, _pipe_zip),
    ("4 workers, uneven task times", "items/s", True, _pipe_uneven),
    ("4 workers, uneven task times, ordered=False", "items/s", True, lambda: _pipe_uneven(ordered=False)),
    ("4MB frames: camera -> sink", "frames/s", True, lambda: _frames("sink")),
    ("4MB frames: camera -> zero-copy sink", "frames/s", True, lambda: _frames("sink", True)),
    ("4MB frames: camera -> 4 workers -> sink", "frames/s", True, lambda: _frames("4 workers")),
    ("4MB frames: camera.tee(2) -> 2 sinks", "frames/s", True, lambda: _frames("tee")),
    ("4MB frames: zip(camera, camera) -> sink", "pairs/s", True, lambda: _frames("zip")),
    ("4MB frames: zip(camera, camera) -> zero-copy sink", "pairs/s", True, lambda: _frames("zip", True)),
]

SUITES = {"queue": CASES, "pipeline": PIPELINE_CASES}

# ------------------------------------------------------------------------------------------ reporting


def _report(results, variants, rounds, cases):
    base, head = variants[0][0], variants[1][0]
    lines = [
        f"| case | unit | {base} | {head} | change | {head} better in |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, unit, higher_better, _ in cases:
        b, h = results[base].get(name, []), results[head].get(name, [])
        if not b or not h:
            continue
        pairs = list(zip(h, b))
        ratios = [x / y for x, y in pairs]
        speedup = statistics.median(ratios) if higher_better else statistics.median(1 / r for r in ratios)
        better = sum((x > y) if higher_better else (x < y) for x, y in pairs)
        fmt = "{:.3g}" if unit in ("GB/s", "ms") else "{:,.1f}" if unit == "us" else "{:,.0f}"
        lines.append(
            f"| {name} | {unit} | {fmt.format(statistics.median(b))} | {fmt.format(statistics.median(h))} "
            f"| {speedup:.2f}x | {better}/{len(pairs)} |"
        )
    lines.append("")
    lines.append(
        f"Medians of {rounds} alternating rounds. *change*: median over rounds of the per-round speed ratio "
        f"{head}/{base} (> 1 means {head} is faster)."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- main


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        for name, unit, _, fn in SUITES[sys.argv[2]]:
            try:
                print(json.dumps({"case": name, "value": fn()}), flush=True)
            except Exception as e:  # report and continue with the other cases
                print(json.dumps({"case": name, "error": repr(e)}), flush=True)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs=2, help="label=path (baseline first)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--markdown", help="also write the report to this file")
    ap.add_argument("--suite", choices=sorted(SUITES), default="queue")
    ap.add_argument("--run-timeout", type=float, default=900, help="seconds per variant and round")
    args = ap.parse_args()
    variants = [tuple(v.split("=", 1)) for v in args.variants]
    results = {label: {} for label, _ in variants}
    for r in range(args.rounds):
        order = variants if r % 2 == 0 else variants[::-1]
        for label, path in order:
            env = dict(os.environ, PYTHONPATH=os.path.abspath(path))
            try:
                proc = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--worker", args.suite], env=env, capture_output=True,
                    text=True, timeout=args.run_timeout,
                )
            except subprocess.TimeoutExpired as e:  # e.g. a hang in an old version: keep what was measured
                out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                proc = subprocess.CompletedProcess(e.cmd, -1, out, f"timed out after {args.run_timeout:.0f} s")
            for line in proc.stdout.splitlines():
                if line.startswith("{"):
                    rec = json.loads(line)
                    if "value" in rec:
                        results[label].setdefault(rec["case"], []).append(rec["value"])
                    else:
                        print(f"round {r + 1} {label}: {rec['case']} failed: {rec['error']}", file=sys.stderr)
            if proc.returncode != 0:
                print(f"round {r + 1} {label}: exit code {proc.returncode}\n{proc.stderr[-3000:]}", file=sys.stderr)
        print(f"round {r + 1}/{args.rounds} done", file=sys.stderr, flush=True)
    report = _report(results, variants, args.rounds, SUITES[args.suite])
    print(report)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
