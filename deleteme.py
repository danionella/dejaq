#!/usr/bin/env python3
import os, sys, time, argparse, pickle, dataclasses
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Any

# Choose a safe default start method
START = "fork" if sys.platform.startswith("linux") else "spawn"
ctx = mp.get_context(START)

@dataclasses.dataclass
class FrameInfo:
    nbytes: int
    head: int
    tail: int
    meta: Any

class ByteFIFO:
    """Ring buffer queue for bytes with backends: 'shm' or 'mparray'."""
    def __init__(self, buffer_bytes: int, backend: str = "shm", use_copyto: bool = False):
        self.buffer_bytes = int(buffer_bytes)
        self.backend = backend
        self.use_copyto = use_copyto
        if backend == "shm":
            self.shm = shared_memory.SharedMemory(create=True, size=self.buffer_bytes)
            self.shm_name = self.shm.name
            self.buffer = self.shm.buf
            self._arr = None
        elif backend == "mparray":
            self.shm = None; self.shm_name = None
            self._arr = ctx.Array('B', self.buffer_bytes, lock=False)
            self.buffer = memoryview(self._arr)
        else:
            raise ValueError("backend must be 'shm' or 'mparray'")
        self._view = None

        self.queue = ctx.Queue()
        self.get_lock = ctx.Lock()
        self.put_lock = ctx.Lock()
        self.head_changed = ctx.Condition()
        self.head = ctx.Value("l", 0)
        self.tail = ctx.Value("l", 0)
        self.closed = ctx.Value("b", False)

    # spawn-safe pickling for SharedMemory
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_view"] = None
        # buffer/memoryview are not picklable
        d.pop("buffer", None)
        return d
    def __setstate__(self, d):
        self.__dict__ = d
        if self.backend == "shm":
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.buffer = self.shm.buf
        else:
            self.buffer = memoryview(self._arr)
        self._view = None

    @property
    def view(self):
        if self._view is None:
            self._view = np.frombuffer(self.buffer, dtype=np.uint8, count=self.buffer_bytes)
        return self._view

    def _available_space(self):
        return (self.head.value - self.tail.value - 1) % self.buffer_bytes

    def _write_buffer(self, array_bytes: np.ndarray):
        tail = self.tail.value
        nbytes = int(array_bytes.size if array_bytes.dtype == np.uint8 else len(array_bytes))
        if tail + nbytes <= self.buffer_bytes:
            if self.use_copyto:
                np.copyto(self.view[tail:tail+nbytes], array_bytes, casting='no')
            else:
                self.view[tail:tail+nbytes] = array_bytes
            self.tail.value = (tail + nbytes) % self.buffer_bytes
        else:
            part = self.buffer_bytes - tail
            if self.use_copyto:
                np.copyto(self.view[tail:], array_bytes[:part], casting='no')
                np.copyto(self.view[:nbytes - part], array_bytes[part:], casting='no')
            else:
                self.view[tail:] = array_bytes[:part]
                self.view[:nbytes - part] = array_bytes[part:]
            self.tail.value = nbytes - part
        return nbytes

    def put(self, array_bytes, meta=None, timeout=None):
        if isinstance(array_bytes, memoryview):
            mv = array_bytes
            nbytes = mv.nbytes
            arr = np.frombuffer(mv, dtype=np.uint8, count=nbytes)
        elif isinstance(array_bytes, np.ndarray):
            arr = np.ascontiguousarray(array_bytes.ravel().view(np.uint8))
            nbytes = arr.size
        else:
            raise TypeError("array_bytes must be memoryview or numpy.ndarray")

        assert nbytes < self.buffer_bytes, "payload larger than ring buffer"
        with self.put_lock:
            while self._available_space() < nbytes:
                with self.head_changed:
                    if not self.head_changed.wait(timeout=timeout):
                        raise TimeoutError("Timeout waiting for space")
            head = self.tail.value
            self._write_buffer(arr)
            self.queue.put(FrameInfo(nbytes=nbytes, head=head, tail=self.tail.value, meta=meta))

    def get(self, callback=None, copy=None, **kwargs):
        with self.get_lock:
            fi = self.queue.get(**kwargs)
            if fi is Ellipsis:
                self.closed.value = True
                return Ellipsis
            head, tail = fi.head, fi.tail
            assert head == self.head.value, f"head mismatch: {head} vs {self.head.value}"
            if head <= tail:
                view = self.view[head:tail]
                if copy or ((copy is None) and (callback is None)):
                    view = np.ascontiguousarray(view)
            else:
                n = fi.nbytes
                out = np.empty(n, dtype=np.uint8)
                part1 = self.view[head:]
                p1 = part1.size
                if self.use_copyto:
                    np.copyto(out[:p1], part1, casting='no')
                    np.copyto(out[p1:], self.view[:tail], casting='no')
                else:
                    out[:p1] = part1
                    out[p1:] = self.view[:tail]
                view = out
            self.head.value = (head + fi.nbytes) % self.buffer_bytes
        with self.head_changed:
            self.head_changed.notify()
        if callback is None:
            return view, fi.meta
        return callback(view, fi.meta)

    def _signal_stop(self, n=1):
        for _ in range(n):
            self.queue.put(Ellipsis)

    def cleanup(self):
        if self.backend == "shm" and getattr(self, "shm", None) is not None:
            try:
                self.shm.close(); self.shm.unlink()
            except FileNotFoundError:
                pass

class DejaQueue(ByteFIFO):
    def put(self, obj, timeout=None):
        buffers = []
        pkl = pickle.dumps(obj, buffer_callback=buffers.append, protocol=pickle.HIGHEST_PROTOCOL)
        lens = [len(pkl)] + [len(b.raw()) for b in buffers]
        nbytes_total = sum(lens)
        assert nbytes_total < self.buffer_bytes, "envelope+buffers exceed ring size"
        with self.put_lock:
            while self._available_space() < nbytes_total:
                with self.head_changed:
                    if not self.head_changed.wait(timeout=timeout):
                        raise TimeoutError("Timeout waiting for space")
            head = self.tail.value
            self._write_buffer(np.frombuffer(pkl, dtype=np.uint8))
            for b in buffers:
                self._write_buffer(b.raw())
            self.queue.put(FrameInfo(nbytes=nbytes_total, head=head, tail=self.tail.value, meta=lens))

    def get(self, **kwargs):
        def cb(byte_vec, lens):
            bufs = []
            off = 0
            for L in lens:
                bufs.append(pickle.PickleBuffer(byte_vec[off:off+L]))
                off += L
            return pickle.loads(bufs[0], buffers=bufs[1:])
        return super().get(copy=False, callback=cb, **kwargs)

def producer(dq: DejaQueue, n_items: int, item_bytes: int):
    arr = np.arange(item_bytes, dtype=np.uint8)
    obj = {"name": "payload", "data": arr}
    for _ in range(n_items):
        dq.put(obj)
    dq._signal_stop(1)

def consumer(dq: DejaQueue):
    while True:
        obj = dq.get()
        if obj is Ellipsis:
            return
        # touch the data to ensure deserialization happens
        _ = obj["data"]

def bench(backend="shm", buffer_mb=256, item_mb=2, n_items=128, use_copyto=False):
    ring_bytes = buffer_mb * 1024 * 1024
    item_bytes = item_mb * 1024 * 1024
    dq = DejaQueue(buffer_bytes=ring_bytes, backend=backend, use_copyto=use_copyto)

    p = ctx.Process(target=producer, args=(dq, n_items, item_bytes))
    c = ctx.Process(target=consumer, args=(dq,))
    t0 = time.perf_counter()
    p.start(); c.start()
    p.join(); c.join()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    dq.cleanup()

    total_payload = n_items * item_bytes  # count array bytes only
    return {
        "backend": backend,
        "copy_path": "np.copyto" if use_copyto else "slicing",
        "items": n_items,
        "item_MB": item_mb,
        "elapsed_s": elapsed,
        "items_per_s": n_items/elapsed,
        "throughput_MBps": total_payload/elapsed/1e6,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["shm","mparray"], default="shm")
    ap.add_argument("--buffer-mb", type=int, default=256)
    ap.add_argument("--item-mb", type=int, default=2)
    ap.add_argument("--n-items", type=int, default=128)
    ap.add_argument("--copyto", action="store_true", help="use np.copyto for ring copies")
    ap.add_argument("--start", choices=["fork","spawn","forkserver"], default=START)
    args = ap.parse_args()

    # re-select context if user requests a different one
    global ctx
    if args.start != START:
        ctx = mp.get_context(args.start)

    res = bench(args.backend, args.buffer_mb, args.item_mb, args.n_items, args.copyto)
    print(res)

if __name__ == "__main__":
    main()
