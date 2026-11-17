"""Transfer speed between two processes vs item size: DejaQueue vs multiprocessing.Queue.

    python benchmarks/size_sweep.py [--out size_sweep] [--label "machine description"]

A child process puts the same uint8 array n times; the main process gets them, timed from the first item
received to the last (process start-up is not timed). Median of 3 runs per size. Writes <out>.json and, if
matplotlib is installed, <out>.png (linear and log speed axes).
"""

import argparse
import json
import multiprocessing as mp
import platform
import statistics
import time

import numpy as np

from dejaq import DejaQueue


def _produce(q, size, n):
    a = np.random.randint(0, 255, size, dtype=np.uint8)
    for _ in range(n):
        q.put(a)


def _speed(make_queue, size):
    n = int(min(5000, max(10, 5e8 // size)))  # ~0.5 GB per run (at least 10 items)
    q = make_queue(size)
    p = mp.get_context("spawn").Process(target=_produce, args=(q, size, n))
    p.start()
    q.get()
    t0 = time.perf_counter()
    for _ in range(n - 1):
        q.get()
    t = time.perf_counter() - t0
    p.join()
    return (n - 1) * size / t / 2**30  # GiB/s


QUEUES = {
    "DejaQueue": lambda size: DejaQueue(max(20e6, 10 * size)),  # a buffer of ~10 items, as recommended
    "mp.Queue": lambda size: mp.get_context("spawn").Queue(),
}
COLORS = {"DejaQueue": "#2a78d6", "mp.Queue": "#eb6834"}


def plot(d, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    for ax, yscale in zip(axes, ("linear", "log")):
        for name, color in COLORS.items():
            ax.plot(d["sizes"], d[name], "o", ms=5, color=color, mec="white", mew=0.8, label=name)
        ax.set(xscale="log", yscale=yscale, xlabel="item size (byte)", ylabel="transfer speed (GiB/s)")
        if yscale == "linear":
            ax.set_ylim(0, None)
        ax.grid(True, color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.text(0.5, 0.01, f"producer and consumer in two processes ({d['label']})", ha="center", color="#52514e",
             fontsize=8)
    plt.tight_layout(rect=(0, 0.03, 1, 0.92))
    plt.savefig(path, facecolor="white")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="size_sweep")
    ap.add_argument("--label", default=f"{platform.system()} {platform.machine()}, Python {platform.python_version()}")
    args = ap.parse_args()
    sizes = [int(s) for s in 10 ** np.arange(3, 8.01, 0.25)]
    d = {"label": args.label, "sizes": sizes}
    for name, make in QUEUES.items():
        d[name] = [statistics.median(_speed(make, s) for _ in range(3)) for s in sizes]
        print(name, " ".join(f"{v:.2f}" for v in d[name]), flush=True)
    with open(args.out + ".json", "w") as f:
        json.dump(d, f)
    try:
        plot(d, args.out + ".png")
    except ImportError:
        print("matplotlib not installed: no plot")
