![Python Version](https://img.shields.io/badge/python-3.8+-blue)
[![PyPI - Version](https://img.shields.io/pypi/v/dejaq)](https://pypi.org/project/dejaq/)
[![Conda Version](https://img.shields.io/conda/v/conda-forge/dejaq)](https://anaconda.org/conda-forge/dejaq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/danionella/dejaq)

# Déjà Queue

A fast alternative to `multiprocessing.Queue`. Faster, because it takes advantage of a shared memory ring buffer (rather than slow pipes) and [pickle protocol 5 out-of-band data](https://peps.python.org/pep-0574/) to minimize copies. [`dejaq.DejaQueue`](#dejaqdejaqueue) supports any type of [picklable](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled) Python object, including numpy arrays or nested dictionaries with mixed content.

<img src="https://github.com/user-attachments/assets/00465436-47f8-4b2a-a236-d288ee34df28" width="100%">

The speed advantage of `DejaQueue` becomes substantial for items of > 1 MB size. It enables efficient inter-job communication in big-data processing pipelines using [`dejaq.Actor`](#dejaqactor-and-actordecorator) or [`dejaq.stream`](#dejaqstream---building-data-pipelines).

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
    while True:
        data = queue.get()
        array, i = data['array'], data['i']
        print(f'consumer {pid} consumed {type(array)} {array.shape} {array.dtype}; index: {i}; hash: {hash(array.tobytes())}\n', flush=True)

queue = DejaQueue(buffer_bytes=100e6)
producer = Process(target=produce, args=(queue,))
consumers = [Process(target=consume, args=(queue, pid)) for pid in range(3)]
for c in consumers:
    c.start()
producer.start()
```

## dejaq.Actor and ActorDecorator

`dejaq.Actor` allows you to run a class instance in a separate process and call its methods or access its attributes remotely, as if it were local. This is useful for isolating heavy computations, stateful services, or legacy code in a separate process, while keeping a simple Pythonic interface.


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

### Features

- **Remote method calls:** Call methods as if the object was local.
- **Remote attribute access:** Get/set attributes of the remote object.
- **Async support:** Call `method_async()` to get a `Future` for non-blocking calls.
- **Tab completion:** Works in Jupyter and most IDEs.


## dejaq.stream - Building Data Pipelines

The `dejaq.stream` module provides a declarative API for building efficient multi-process data pipelines. Each pipeline stage is a “node”, and nodes run their work in separate process(es), communicating through fast `DejaQueue`-backed channels.

You can build nodes from either **functions** (executed in worker processes for each item) or **classes** (instantiated once inside a worker process, then called remotely for each item). This makes it easy to compose stateful processors (classes) and stateless transforms (functions) in the same pipeline.


### Simple self-explanatory example:


```python
from dejaq.stream import Source
import numpy as np
from scipy.ndimage import gaussian_filter


class CameraController:
    def get_frame(self):
        return np.random.randn(480, 640)

class GaussianSmoother:
    def __init__(self, sigma=2.0):
        self.sigma = sigma
        self.count = 0
    
    def __call__(self, frame):
        self.count += 1
        return gaussian_filter(frame, sigma=self.sigma)

# Create a source that generates random frames at 30 fps
src = Source(cls=CameraController, call_fcn=lambda cam: cam.get_frame(), rate=30)

# Build a pipeline: preprocess -> detect -> save
    
pipeline = (
    src 
    .map(fcn = lambda frame: (frame - frame.min()) / (frame.max() - frame.min()), n_workers=4)  # normalize
    .map(cls = lambda: GaussianSmoother(sigma=3.0))  # smooth with gaussian filter
    .sink(fcn = lambda frame: print(f"Processed frame: mean={frame.mean():.3f}, std={frame.std():.3f}"))
)

# Start the source
src.start()

# Stop after some time
import time
time.sleep(5)
src.stop()
```

> [!IMPORTANT]
> Keep a reference to all source nodes. A bare expression like `Source(...).map(...).sink(...)` with no assignment can be garbage-collected immediately.

### API Reference

#### `Source(it=None, fcn=None, cls=None, call_fcn=..., init_kwargs=None, rate=None, ...)`

Create a source node from an iterable, function, or class instance:

```python
Source(it=range(100))                                        # from iterable
Source(fcn=lambda: get_data(), rate=30)                      # from function, rate-limited to 30 Hz
Source(cls=Camera, call_fcn=lambda c: c.get_frame())         # from class instance
Source()                                                     # manual source (use .put(some_data) and .stop())
```

#### `.map(fcn=None, cls=None, cls_fcn=..., init_kwargs=None, n_workers=1, ...)`

Apply a function or class to each item:

```python
node.map(fcn=lambda x: x * 2, n_workers=4)                   # function with 4 workers
node.map(cls=Processor)                                      # class (calls .__call__ on each item)
node.map(cls=lambda: Proc(x=5), cls_fcn=lambda p, x: p.process(x))      # calls method "process" on each item
```

#### `.tee(count=2)` and `.zip(*nodes)`

Split and combine streams:

```python
stream1, stream2 = node.tee(count=2)                         # split into 2 independent streams
stream1_processed = stream1.map(fcn=lambda x: x * 2)
recombined = stream1_processed.zip(stream2)     # yields (item1, item2) tuples
```

> [!IMPORTANT]
> When working with multiple sources or split streams, make sure to eventually consume all outputs.


#### `.tqdm()`

Add a progress bar:

```python
node = node.tqdm(desc="Processing items", total=1000)
```


#### `.sink(fcn=None, factory=None, ...)` and `.run()`

Consume the stream:

```python
# using sink nodes (source is typically started after creating the sink)
pipeline.sink(fcn=lambda x: print(x))                            # terminal node, no output
# ...
src.start()

# using .run() to collect results (source should be started before calling .run())
src.start()
results = pipeline.run()                                         # collect all results (blocking)
```
> [!IMPORTANT]
> When collecting results using `.run()` (blocking call), make sure the upstream sources have already been started (e.g. with `src.start()`). The pattern is to first start all sources, then call `.run()` on the terminal node(s). For `.sink()`, the source is usually started after creating the sink node.

#### Control methods

```python
src.start()          # start a source (required for Source nodes)
src.stop()           # signal cancellation
node.is_running()    # check if node's workers are alive
```

### Advanced Example with multiple branches and custom classes

This example shows how to branch a stream with `tee()`, process each branch differently (one branch using a plain function, the other using a stateful class instantiated in a worker), and then recombine branches with `zip()`.

```python
import numpy as np
from dejaq.stream import Source


class RunningMean:
    """Stateful processor: instantiated once inside a worker."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0

    def __call__(self, x: float) -> dict:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        return {"x": x, "mean": self.mean}


# Finite input source
rng = np.random.default_rng(0)
src = Source(it=rng.normal(size=1000))

# Branch the stream
a, b = src.tee(2)

# Branch A: stateless function (runs per-item in worker processes)
abs_x = a.map(fcn=lambda x: float(abs(x)), n_workers=4)

# Branch B: stateful class (instantiated in a worker, then called per item)
stats = b.map(cls=RunningMean, n_workers=1)

# Recombine: primary drives output timing
joined = abs_x.zip(stats, mode="sync")

# Terminal sink: consume the stream (runs eagerly)
sink = joined.sink(fcn=lambda pair: None)

# Wait for the sink workers to finish consuming the finite source
sink.wait()
```


<!--
### dejaq.Parallel
The following examples show how to use `dejaq.Parallel` to parallelize a function or a class, and how to create job pipelines.

Here we execute a function and map iterable inputs across 10 workers. To enable pipelining, the results of each stage are provided as iterable generator. Use `.run()` (or `.compute()` for backwards compatibility) to get the final result. Results are always ordered.

```python
from time import sleep
from dejaq import Parallel

def slow_function(arg):
    sleep(1.0)
    return arg + 5

input_iterable = range(100)
slow_function = Parallel(n_workers=10)(slow_function)
stage = slow_function(input_iterable)
result = stage.run() # or list(stage)
# or shorter: 
result = Parallel(n_workers=10)(slow_function)(input_iterable).compute()
```

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
-->

# See also
- [ArrayQueues](https://github.com/portugueslab/arrayqueues) 
- [joblib.Parallel](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
- [Déjà Q](https://en.wikipedia.org/wiki/Deja_Q)
