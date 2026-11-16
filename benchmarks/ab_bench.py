"""A/B benchmark of DejaQueue: compares two checkouts of dejaq on the same machine.

    python benchmarks/ab_bench.py --rounds 5 base=<path to checkout> head=<path to checkout> [--markdown out.md]

Each round runs every case once per variant, each variant in a fresh process with that checkout on
PYTHONPATH; the order of the variants alternates between rounds, so slow drifts of a (shared) machine
affect both alike. The report gives medians and per-round paired ratios (head / base, for times).
Only the public DejaQueue API is used, so any dejaq version can be compared.
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

# ------------------------------------------------------------------------------------------ reporting


def _report(results, variants, rounds):
    base, head = variants[0][0], variants[1][0]
    lines = [
        f"| case | unit | {base} | {head} | change | {head} better in |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, unit, higher_better, _ in CASES:
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
        for name, unit, _, fn in CASES:
            try:
                print(json.dumps({"case": name, "value": fn()}), flush=True)
            except Exception as e:  # report and continue with the other cases
                print(json.dumps({"case": name, "error": repr(e)}), flush=True)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs=2, help="label=path (baseline first)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--markdown", help="also write the report to this file")
    args = ap.parse_args()
    variants = [tuple(v.split("=", 1)) for v in args.variants]
    results = {label: {} for label, _ in variants}
    for r in range(args.rounds):
        order = variants if r % 2 == 0 else variants[::-1]
        for label, path in order:
            env = dict(os.environ, PYTHONPATH=os.path.abspath(path))
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker"], env=env, capture_output=True, text=True
            )
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
    report = _report(results, variants, args.rounds)
    print(report)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
