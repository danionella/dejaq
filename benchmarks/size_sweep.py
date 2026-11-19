"""Transfer speed between two processes vs item size: DejaQueue vs multiprocessing.Queue.

    python benchmarks/size_sweep.py [--out size_sweep] [--label "machine description"]

A child process puts the same uint8 array n times; the main process gets them, timed after a first lap that
fills the whole ring buffer once (process start-up and first-touch page faults of the shared memory are not
timed). Median of 3 runs per size. DejaQueue is read three ways:
get() (a new array per item), a copy into one preallocated array (get(callback=...)), and zero-copy (the
callback receives a view of the queue's memory and copies nothing). Also records what limits a copy-based
queue: the CPU, cores, caches, RAM, and how fast one core copies memory (in cache and from RAM). Writes
<out>.json and, if matplotlib is installed, <out>.png (linear and log speed axes).
"""

import argparse
import json
import multiprocessing as mp
import platform
import statistics
import subprocess
import time

import numpy as np

from dejaq import DejaQueue


def _produce(q, size, n):
    a = np.random.randint(0, 255, size, dtype=np.uint8)
    for _ in range(n):
        q.put(a)


def _speed(make_queue, read, size):
    n = int(min(5000, max(10, 5e8 // size)))  # ~0.5 GB per run (at least 10 items)
    q, buf = make_queue(size), np.empty(size, np.uint8)
    warm = getattr(q, "cap", 0) // size + 2  # untimed: a lap through the whole ring (DejaQueue), and one item
    get = {"copy": q.get, "into": lambda: q.get(callback=lambda a: np.copyto(buf, a)),
           "none": lambda: q.get(callback=lambda a: None)}[read]
    p = mp.get_context("spawn").Process(target=_produce, args=(q, size, n + warm - 1))
    p.start()
    for _ in range(warm):
        get()
    t0 = time.perf_counter()
    for _ in range(n - 1):
        get()
    t = time.perf_counter() - t0
    p.join()
    return (n - 1) * size / t / 2**30  # GiB/s


def _copy_speed(nbytes):
    """GiB/s of one core copying nbytes (np.copyto), median of 5 after a warm-up copy."""
    a, b = np.ones(nbytes, np.uint8), np.empty(nbytes, np.uint8)
    np.copyto(b, a)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        np.copyto(b, a)
        times.append(time.perf_counter() - t0)
    return nbytes / statistics.median(times) / 2**30


def _machine():
    """What limits a copy-based queue: CPU, cores, caches, RAM, one core's copy speed (best effort per OS)."""
    import psutil

    m = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(), "numpy": np.__version__,
         "cpu": platform.processor() or platform.machine(), "cores": psutil.cpu_count(logical=False),
         "threads": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)}
    try:
        if platform.system() == "Linux":
            m["cpu"] = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
            for i in (2, 3):
                with open(f"/sys/devices/system/cpu/cpu0/cache/index{i}/size") as f:
                    m[f"l{i}_cache"] = f.read().strip()
        elif platform.system() == "Darwin":
            sysctl = lambda key: subprocess.check_output(["sysctl", "-n", key], text=True).strip()
            m["cpu"] = sysctl("machdep.cpu.brand_string")
            m["l2_cache"] = f"{int(sysctl('hw.l2cachesize')) // 1024}K"
        elif platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            m["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:  # (whatever could be found)
        pass
    m["copy_gib_s_2MB"] = round(_copy_speed(2 << 20), 1)  # (in cache)
    m["copy_gib_s_256MB"] = round(_copy_speed(256 << 20), 1)  # (from RAM)
    return m


def _machine_text(m):
    caches = ", ".join(f"{k[:2].upper()} {int(m[k].rstrip('K')) / 1024:g} MB" for k in ("l2_cache", "l3_cache")
                       if k in m)
    return (f"{m['cpu']}, {m['cores']} cores / {m['threads']} threads{', ' + caches if caches else ''}, "
            f"{m['ram_gb']:g} GB RAM\none core copies {m['copy_gib_s_2MB']:g} GiB/s (2 MB, in cache), "
            f"{m['copy_gib_s_256MB']:g} GiB/s (256 MB, from RAM); {m['os']}, Python {m['python']}, numpy {m['numpy']}")


def _dejaq(size):
    return DejaQueue(max(20e6, 10 * size))  # a buffer of ~10 items, as recommended


QUEUES = {  # name: (queue factory, how items are read, color, marker)
    "DejaQueue": (_dejaq, "copy", "#2a78d6", "o"),
    "mp.Queue": (lambda size: mp.get_context("spawn").Queue(), "copy", "#eb6834", "s"),
    "DejaQueue, into preallocated array": (_dejaq, "into", "#1baf7a", "^"),
    "DejaQueue, zero-copy": (_dejaq, "none", "#eda100", "D"),
}


def plot(d, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    for ax, yscale in zip(axes, ("linear", "log")):
        for name, (_, _, color, marker) in QUEUES.items():
            if name in d:
                ax.plot(d["sizes"], d[name], marker, ms=5, color=color, mec="white", mew=0.8, label=name)
        ax.set(xscale="log", yscale=yscale, xlabel="item size (byte)", ylabel="transfer speed (GiB/s)")
        if yscale == "linear":
            ax.set_ylim(0, None)
        ax.grid(True, color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=8)
    caption = f"producer and consumer in two processes ({d['label']})"
    if "machine" in d:
        caption += "\n" + _machine_text(d["machine"])
    fig.text(0.5, 0.01, caption, ha="center", color="#52514e", fontsize=7)
    plt.tight_layout(rect=(0, 0.1 if "machine" in d else 0.03, 1, 0.92))
    plt.savefig(path, facecolor="white")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="size_sweep")
    ap.add_argument("--label", default=f"{platform.system()} {platform.machine()}, Python {platform.python_version()}")
    args = ap.parse_args()
    sizes = [int(s) for s in 10 ** np.arange(3, 8.01, 0.25)]
    d = {"label": args.label, "machine": _machine(), "sizes": sizes}
    print(_machine_text(d["machine"]).replace("\n", "; "), flush=True)
    for name, (make, read, _, _) in QUEUES.items():
        d[name] = [statistics.median(_speed(make, read, s) for _ in range(3)) for s in sizes]
        print(name, " ".join(f"{v:.2f}" for v in d[name]), flush=True)
    with open(args.out + ".json", "w") as f:
        json.dump(d, f)
    try:
        plot(d, args.out + ".png")
    except ImportError:
        print("matplotlib not installed: no plot")
