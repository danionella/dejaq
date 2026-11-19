![Python Version](https://img.shields.io/badge/python-3.8+-blue)
[![PyPI - Version](https://img.shields.io/pypi/v/dejaq)](https://pypi.org/project/dejaq/)
[![Conda Version](https://img.shields.io/conda/v/conda-forge/dejaq)](https://anaconda.org/conda-forge/dejaq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/danionella/dejaq)

# Déjà Queue

A fast alternative to `multiprocessing.Queue`. Faster, because it takes advantage of a shared memory ring buffer (rather than slow pipes) and [pickle protocol 5 out-of-band data](https://peps.python.org/pep-0574/) to minimize copies. [`dejaq.DejaQueue`](#dejaqdejaqueue) supports any type of [picklable](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled) Python object, including numpy arrays or nested dictionaries with mixed content.

<img src="https://github.com/user-attachments/assets/837ca980-b225-45b7-8bf5-f0ce6cf7083c" width="100%">

The speed advantage of `DejaQueue` becomes substantial for items of > 1 MB size. It enables efficient inter-job communication in big-data processing pipelines using [`dejaq.Actor`](#dejaqactor-and-actordecorator) or [`dejaq.Parallel`](#dejaqparallel).

### Features:
- Fast, low-latency, high-throughput inter-process communication
- Supports any picklable Python object, including numpy arrays and nested dictionaries
- Zero-copy data transfer with pickle protocol 5 out-of-band data
- Picklable queue instances (queue object itself can be passed between processes)
- Peekable (non-destructive read)
- Actor class for remote method calls and attribute access in a separate process (see [dejaq.Actor](#dejaqactor-and-actordecorator))

Auto-generated (minimal) API documentation: https://danionella.github.io/dejaq


## Installation
- `conda install conda-forge::dejaq `

- or, if you prefer pip: `pip install dejaq`

- for development, clone this repository, navigate to the root directory and type `pip install -e .`

dejaq supports Python 3.8 and later on Linux, macOS and Windows. On free-threaded Python builds (e.g. 3.14t), dejaq works, but on Linux and macOS importing it turns the GIL back on, because its dependency `posix_ipc` does not declare free-threading support yet (see #38).

## Examples
### dejaq.DejaQueue
```python
import numpy as np
from multiprocessing import Process
from dejaq import DejaQueue

def produce(queue):
    for i in range(10):
        arr = np.random.randn(100,200,300)
        data = dict(array=arr, i=i)
        queue.put(data)
        print(f'produced {type(arr)} {arr.shape} {arr.dtype}; meta: {i}; hash: {hash(arr.tobytes())}\n', flush=True)

def consume(queue, pid):
    for data in queue:  # until the queue is shut down and empty
        array, i = data['array'], data['i']
        print(f'consumer {pid} consumed {type(array)} {array.shape} {array.dtype}; index: {i}; hash: {hash(array.tobytes())}\n', flush=True)

if __name__ == "__main__":
    queue = DejaQueue(buffer_bytes=100e6)
    producer = Process(target=produce, args=(queue,))
    consumers = [Process(target=consume, args=(queue, pid)) for pid in range(3)]
    for c in consumers:
        c.start()
    producer.start()
    producer.join()
    queue.shutdown()  # consumers finish the remaining items, then stop
    for c in consumers:
        c.join()
```

`DejaQueue` follows the interface of `queue.Queue` / `multiprocessing.Queue`:
- `put(obj, block=True, timeout=None)`, `get(block=True, timeout=None)`, `put_nowait(obj)`, `get_nowait()`, `qsize()`, `empty()`, `full()`
- `shutdown(immediate=False)` (as in Python 3.13): afterwards `put()` raises `ShutDown`, and `get()` returns the remaining items and then raises `ShutDown`; this applies to all processes using the queue, and iterating over a queue ends there. With `immediate=True`, remaining items are dropped.
- Exceptions: `dejaq.Empty`, `dejaq.Full`, `dejaq.ShutDown`, which are also `queue.Empty`, `queue.Full` (both also `TimeoutError`) and, on Python ≥ 3.13, `queue.ShutDown`
- When the queue is full, `put(obj, on_full=...)` chooses what happens: `"block"` (the default: wait, as set by `block` and `timeout`), `"drop"` (discard `obj`; `put` returns `False`), `"drop_oldest"` (discard the oldest items until `obj` fits, e.g. to keep the newest camera frames when the consumer falls behind) or `"flush"` (discard all queued items). `q.dropped` counts the items that this queue object's `put()` discarded. With a buffer that holds only one item (e.g. `buffer_bytes=1.5 * frame.nbytes`), `"drop_oldest"` keeps only the latest item. A stop item (e.g. `None`) can be discarded as well: end the stream with `shutdown()` instead.
- Additionally: `get(peek_only=True)` reads an item without removing it (a later `get()` may get another item if a producer with `on_full="drop_oldest"` or `"flush"` discarded it meanwhile), and `get(callback=fn)` calls `fn` on the item while its numpy arrays still view the shared memory (zero-copy; `fn` must not keep references to them)
- A queue object can be pickled and passed to other processes, including independently started ones (`DejaQueue(name=..., create=False)` attaches by name). It is removed once the creating object is garbage-collected, unless created with `auto_unlink=False`. It can also be used as a context manager.
- It keeps working when a process using it dies, even while putting or getting an item.

## dejaq.Actor and ActorDecorator

`dejaq.Actor` allows you to run a class instance in a separate process and call its methods or access its attributes remotely, as if it were local. This is useful for isolating heavy computations, stateful services, or legacy code in a separate process, while keeping a simple Pythonic interface.

Actors start their process with the `spawn` method, which imports your main script in it: in a script (unlike in Jupyter), put the code that creates actors under `if __name__ == "__main__":`, as in the DejaQueue example above.

### Example: Using `Actor` directly

```python
from dejaq import Actor

class Counter:
    def __init__(self, start=0):
        self.value = start
    def increment(self, n=1):
        self.value += n
        return self.value
    def get(self):
        return self.value

# Start the actor in a separate process
counter = Actor(Counter, start=10)

print(counter.get())         # 10
print(counter.increment())   # 11
print(counter.increment(5))  # 16
print(counter.get())         # 16

counter.close()  # Clean up the process
```

### Example: Using `ActorDecorator`

```python
from dejaq import ActorDecorator

@ActorDecorator
class Greeter:
    def __init__(self, name):
        self.name = name
    def greet(self):
        return f"Hello, {self.name}!"

greeter = Greeter("Alice")
print(greeter.greet())  # "Hello, Alice!"
greeter.close()
```

Use `with Actor(Counter) as counter:` to close it automatically. An actor is also stopped when the `Actor` object is garbage-collected, and when your program exits. On stopping, the remote object's `close()` method is called, if it has one (the process is terminated if it has not ended after 10 s, see `close(timeout=...)`).

### Features

- **Remote method calls:** Call methods as if the object was local. Exceptions raised by the remote method are raised again in your process, with the same type (and the remote traceback attached).
- **Remote attribute access:** Get and set attributes of the remote object (`counter.value`, `counter.value = 5`).
- **Async calls:** `counter.increment_async(5)` returns a `Future` (`.result()` waits for the value). `counter.increment(5, noreply=True)` does not wait at all (the result is discarded). With `noreply=True, on_full="drop"`, the call is skipped (and returns `False`) when the actor is too busy to take it, e.g. to send camera frames to a display actor without slowing the camera down.
- **Timeouts:** `counter.get(timeout=2)` raises `TimeoutError` if there is no result within 2 s. If the actor's process has ended (it crashed or was closed), calls raise `dejaq.remote.RemoteError` instead of waiting.
- **Threads and worker processes:** An actor can be used from several threads, and passed to other processes (e.g. as an argument of a `Parallel` function).
- **Zero-copy fast path:** Pass `deepcopy=False` to skip the default copy of numpy array arguments (e.g. `actor.process(arr, deepcopy=False)`). The method then reads the arrays directly in shared memory, which is only valid during the call: copy anything you want to keep (`self.last = arr.copy()`); returning a result computed from them is fine. A `RuntimeWarning` is shown if a method keeps a reference.
- **Tab completion:** Works in Jupyter and most IDEs.

### Running a function in worker processes: `RemoteFunc`

```python
from dejaq import RemoteFunc

def square(x):
    return x * x

with RemoteFunc(square, workers=4) as rf:
    print(rf(12))               # 144
    print(rf.map(range(5)))     # [0, 1, 4, 9, 16]
    print(rf.submit(7).result())  # 49
```

Notes:
- Each process that uses an actor (or a `RemoteFunc`) gets a reply queue of `buffer_bytes` (default 10 MB) in shared memory. With many worker processes, choose a smaller `buffer_bytes` if the results are small.
- Arguments must be importable by the actor's process. In a notebook, a class defined in the notebook cannot be passed to an actor; such a call waits until its `timeout` (an error is logged by the actor).


## dejaq.Parallel
The following examples show how to use `dejaq.Parallel` to parallelize a function or a class, and how to create job pipelines.

Here we execute a function and map iterable inputs across 10 workers. To enable pipelining, the results of each stage are provided as iterable generator. Use `.run()` (or `.compute()`) to get the final result as a list. Results are in input order, unless you pass `ordered=False` (e.g. `Parallel(n_workers=10, ordered=False)`): then each result is returned as soon as it is ready, so that a slow item does not hold up the others.

Workers are started with the `spawn` method, which imports your main script in each worker: in a script (unlike in Jupyter), put the code that starts pipelines under `if __name__ == "__main__":`.

```python
from time import sleep
from dejaq import Parallel

def slow_function(arg):
    sleep(1.0)
    return arg + 5

input_iterable = range(100)
parallel_function = Parallel(n_workers=10)(slow_function)
stage = parallel_function(input_iterable)
result = stage.run() # or list(stage)
# or shorter: 
result = Parallel(n_workers=10)(slow_function)(input_iterable).run()
```

If a worker raises an exception, `run()` (or iterating) raises the same exception, with the worker's traceback as its cause; if a worker process dies, it raises `dejaq.parallel.WorkerError`. A stage's workers are stopped when that happens, when the stage is closed (`stage.close()` or a `with` block; this also closes the stages that feed it) or garbage-collected, and at exit. Leaving a loop early (`break`) does not stop them: iterating again continues with the next result.

You can also use `Parallel` as a function decorator:
```python
@Parallel(n_workers=10)
def slow_function_decorated(arg):
    sleep(1.0)
    return arg + 5

result = slow_function_decorated(input_iterable).run()
```

Similarly, you can decorate a class. It will be instantiated within a worker. Iterable items will be fed to the `__call__` method. Note how the additional init arguments are provided:
```python
@Parallel(n_workers=1)
class Reader:
    def __init__(self, arg1):
        self.arg1 = arg1
    def __call__(self, item):
        return item + self.arg1

result = Reader(arg1=0.5)(input_iterable).compute()
```

Finally, you can create pipelines of chained jobs. In this example, we have a single threaded reader and consumer, but a parallel processing stage (an example use case is sequentially reading a file, compressing chunks in parallel and then sequentially writing to an output file):
```python
@Parallel(n_workers=1)
class Producer:
    def __init__(self, arg1):
        self.arg1 = arg1
    def __call__(self, item):
        return item + self.arg1

@Parallel(n_workers=10)
class Processor:
    def __init__(self, arg1):
        self.arg1 = arg1
    def __call__(self, arg):
        sleep(1.0) #simulating a slow function
        return arg * self.arg1

@Parallel(n_workers=1)
class Consumer:
    def __init__(self, arg1):
        self.arg1 = arg1
    def __call__(self, arg):
        return arg - self.arg1

input_iterable = range(100)
stage1 = Producer(0.5)(input_iterable)
stage2 = Processor(10.0)(stage1)
stage3 = Consumer(1000)(stage2)
result = stage3.run()

# or:
result = Consumer(1000)(Processor(10.0)(Producer(0.5)(input_iterable))).run()
```

For large items such as camera frames (several MB each, at kHz rates), a few settings matter:
- `buffer_bytes`: each stage's output queue. Make it at least about 10 times the item size, so that the stages can absorb each other's hiccups (a larger buffer does not make the steady state faster).
- `zero_copy=True` (stages with `n_workers=1`, e.g. a writer): the function receives each item from the input stage without a copy, its numpy arrays viewing the queue's memory. They are valid only during the call: copy what you need to keep (e.g. a previous frame), but returning them is fine. Likewise, `for frame in stage.iter(zero_copy=True)` in the main process, where each item is valid until the next one is requested. Zero-copy arrays may be unaligned in memory (e.g. float64 data not at a multiple of 8 bytes), which makes some numpy operations slower; copy them where that matters or a library needs aligned data.
- `stage.qsize()` tells how many results wait in a stage's output queue: one that stays near full means the next stage does not keep up. Stages themselves never drop items; to drop frames when the pipeline falls behind, put them into a `DejaQueue` with `on_full="drop_oldest"` (see above) and pass the queue as the pipeline's input (`Parallel(...)(fcn)(queue)`; it ends when the queue is shut down).
- Each hop between processes copies an item once into a queue and once out of it, so fewer stages mean less copying; `tee` and `zip` copy each item only once. (Measured with 4 MB frames on CI runners: `zip` is 1.2-1.5x faster than with two copies on Windows and macOS, but 0.7x on Linux, for reasons not yet understood.)

A stage's results can be split into several branches with `stage.tee(n)`, and stages can be combined with `dejaq.parallel.zip(stage_a, stage_b)` (which yields tuples). Like the stages themselves, both run in processes of their own, so the data does not pass through your main process. As with `itertools.tee`, consume the branches side by side: a branch that falls behind by more than its buffer holds makes the others wait.


# See also
- [ArrayQueues](https://github.com/portugueslab/arrayqueues) 
- [joblib.Parallel](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
- [Déjà Q](https://en.wikipedia.org/wiki/Deja_Q)
