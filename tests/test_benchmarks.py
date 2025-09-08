# test_named_semaphore_bench.py
"""
Pytest benchmarks for NamedSemaphore.

What it measures:
1) Uncontended acquire+release pairs (same process)
2) Half-hop ping–pong latency (parent <-> child)
3) Burst throughput (producer releases N, consumer drains N)

Prints benchmark results and asserts only lenient sanity bounds
(to avoid flakiness across machines/CI).

Tips:
- To see printed lines with pytest, run:  pytest -q -s
- Scale down/up with environment variables:
    NSEM_FAST=1      -> fewer iterations (quicker)
    NSEM_ROUNDS=...  -> override ping–pong rounds (int)
    NSEM_TOKENS=...  -> override burst tokens (int)
"""
from __future__ import annotations
import os
import time
import statistics
import multiprocessing as mp
import pytest

# ---- import your NamedSemaphore here ----
try:
    from dejaq.queues import NamedSemaphore  # <-- change if your module name differs
except Exception as e:  # pragma: no cover
    pytest.skip(f"NamedSemaphore import failed: {e}", allow_module_level=True)


# ---------------- helpers ----------------
def _get_ctx_name() -> str:
    # macOS/Windows default to 'spawn'; Linux can use 'fork' for slightly lower overhead
    return os.environ.get("NSEM_CTX", "spawn")

def _iters(default_fast: int, default_full: int) -> int:
    return int(os.environ.get("NSEM_ITERS", default_fast if os.environ.get("NSEM_FAST") else default_full))


# ---------------- 1) Uncontended ----------------
def bench_uncontended(iterations: int = 200_000) -> dict:
    sem = NamedSemaphore(create=True, initial=1)
    # warmup
    for _ in range(min(5_000, iterations // 10)):
        sem.acquire(); sem.release()
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        sem.acquire(); sem.release()
    t1 = time.perf_counter_ns()
    sem.close(); sem.unlink()
    pair_ns = (t1 - t0) / iterations
    return {"iterations": iterations, "pair_ns": pair_ns, "pair_us": pair_ns / 1e3}


# ---------------- 2) Ping–pong (half-hop) ----------------
def _child_ping(gate_a_name: str, gate_b_name: str, iters: int, ready):
    a = NamedSemaphore(name=gate_a_name, create=False)
    b = NamedSemaphore(name=gate_b_name, create=False)
    ready.send("ready")
    for _ in range(iters):
        a.acquire()
        b.release()

def bench_ping_pong(rounds: int = 10_000, warmup: int = 1_000, ctx_name: str = "spawn") -> dict:
    ctx = mp.get_context(ctx_name)
    gate_a = NamedSemaphore(create=True, initial=0)
    gate_b = NamedSemaphore(create=True, initial=0)
    pa, pb = ctx.Pipe(duplex=False)
    iters = warmup + rounds
    p = ctx.Process(target=_child_ping, args=(gate_a.name, gate_b.name, iters, pb))
    p.start()
    pa.recv()  # child ready

    half_us = []
    for i in range(iters):
        t0 = time.perf_counter_ns()
        gate_a.release()
        gate_b.acquire()
        if i >= warmup:
            half_us.append(((time.perf_counter_ns() - t0) / 2.0) / 1e3)

    p.join()
    gate_a.close(); gate_a.unlink()
    gate_b.close(); gate_b.unlink()
    return {
        "rounds": rounds,
        "median_us": statistics.median(half_us),
        "p90_us": statistics.quantiles(half_us, n=10)[8] if len(half_us) >= 10 else max(half_us),
        "min_us": min(half_us),
        "max_us": max(half_us),
    }


# ---------------- 3) Burst throughput ----------------
def _child_drain(gate_name: str, count: int, ready, done):
    g = NamedSemaphore(name=gate_name, create=False)
    ready.send("ready")
    for _ in range(count):
        g.acquire()
    done.send("done")

def bench_burst(tokens: int = 200_000, ctx_name: str = "spawn") -> dict:
    ctx = mp.get_context(ctx_name)
    gate = NamedSemaphore(create=True, initial=0)
    pa_ready, pb_ready = ctx.Pipe(duplex=False)
    pa_done,  pb_done  = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child_drain, args=(gate.name, tokens, pb_ready, pb_done))
    p.start()
    pa_ready.recv()

    t0 = time.perf_counter_ns()
    for _ in range(tokens):
        gate.release()
    pa_done.recv()
    t1 = time.perf_counter_ns()

    p.join()
    gate.close(); gate.unlink()

    elapsed_s = (t1 - t0) / 1e9
    return {
        "tokens": tokens,
        "elapsed_s": elapsed_s,
        "throughput_Mops": tokens / elapsed_s / 1e6,
        "avg_per_token_us": (elapsed_s / tokens) * 1e6,
    }


# ======================= pytest tests (print + assert) =======================

def test_named_semaphore_uncontended():
    iters = _iters(default_fast=50_000, default_full=200_000)
    res = bench_uncontended(iters)
    print(f"[uncontended] iterations={res['iterations']:,}  "
          f"pair={res['pair_us']:.3f} µs")
    # Sanity: operations complete and not absurdly slow on a sane system
    assert res["pair_us"] > 0
    assert res["pair_us"] < 50_000  # 50 ms per pair would indicate a problem


def test_named_semaphore_ping_pong():
    rounds = int(os.environ.get("NSEM_ROUNDS", 5_000 if os.environ.get("NSEM_FAST") else 20_000))
    res = bench_ping_pong(rounds=rounds, warmup=max(200, rounds // 10), ctx_name=_get_ctx_name())
    print(f"[ping-pong]   rounds={res['rounds']:,}  "
          f"median={res['median_us']:.3f} µs  p90={res['p90_us']:.3f} µs  "
          f"min={res['min_us']:.3f} µs  max={res['max_us']:.3f} µs")
    # Sanity: latencies should be finite and not outrageous
    assert 0 < res["median_us"] < 50_000
    assert 0 < res["p90_us"]   < 100_000


def test_named_semaphore_burst():
    tokens = int(os.environ.get("NSEM_TOKENS", 50_000 if os.environ.get("NSEM_FAST") else 200_000))
    res = bench_burst(tokens=tokens, ctx_name=_get_ctx_name())
    print(f"[burst]       tokens={res['tokens']:,}  "
          f"elapsed={res['elapsed_s']:.3f}s  "
          f"throughput={res['throughput_Mops']:.3f} Mops  "
          f"avg={res['avg_per_token_us']:.3f} µs/token")
    # Sanity: positive throughput and reasonable per-token time
    assert res["throughput_Mops"] > 0
    assert 0 < res["avg_per_token_us"] < 100_000


