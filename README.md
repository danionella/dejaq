
![Python Version](https://img.shields.io/badge/python-3.8+-blue)
[![PyPI - Version](https://img.shields.io/pypi/v/dejaq)](https://pypi.org/project/dejaq/)
[![Conda Version](https://img.shields.io/conda/v/danionella/dejaq)](https://anaconda.org/danionella/dejaq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/danionella/dejaq)

# Déjà Queue

A drop-in replacement for `multiprocessing.Queue`. Faster, because it takes advantage of a shared memory ring buffer (rather than slow pipes) and [pickle protocol 5 out-of-band data](https://peps.python.org/pep-0574/) to minimize copies. `DejaQueue` supports any type of [picklable](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled) Python object, including numpy arrays or nested dictionaries with mixed content.

The speed of `DejaQueue` enables efficient inter-job communication in data processing pipelines, which can be implemented in a few lines of code with [`dejaq.Parallel`](#dejaqparallel).

Auto-generated API documentation: https://danionella.github.io/dejaq

## Examples
### `dejaq.DejaQueue`
```python
import numpy as np
from multiprocessing import Process
from dejaq import DejaQueue

def produce(queue):
    for i in range(20):
        random_shape = np.random.randint(5,10, size=3)
        array = np.random.randn(*random_shape)
        queue.put(array, meta=i)
        print(f'produced {type(array)} {array.shape} {array.dtype}; meta: {i}; hash: {hash(array.tobytes())}\n')

def consume(queue, pid):
    while True:
        array, meta = queue.get()
        print(f'consumer {pid} consumed {type(array)} {array.shape} {array.dtype}; meta: {meta}; hash: {hash(array.tobytes())}\n')

queue = DejaQueue(bytes=10e6)
producer = Process(target=produce, args=(queue,))
consumers = [Process(target=consume, args=(queue, pid)) for pid in range(3)]
for c in consumers:
    c.start()
producer.start()
```


### `dejaq.Parallel`
The following examples show how to use `dejaq.Parallel` to parallelize a function or a class, and how to create job pipelines.

Here we execute a function and map iterable inputs across 10 workers. To enable pipelining, the results of each stage are provided as iterable generator. Use the `.compute()` method to get the final result (note that each stage pre-fetches results from `n_workers` calls, so some of the execution already starts before `.compute`)

```python
from time import sleep
from dejaq import Parallel

def slow_function(arg):
    sleep(1.0)
    return arg + 5

input_iterable = range(100)
slow_function = Parallel(n_workers=10)(slow_function)
stage = slow_function(input_iterable)
result = stage.compute() # or list(stage)
# or shorter: 
result = Parallel(n_workers=10)(slow_function)(input_iterable).compute()
```

You can also use `Parallel` as a function decorator:
```python
@Parallel(n_workers=10)
def slow_function_decorated(arg):
    sleep(1.0)
    return arg + 5

result = slow_function_decorated(input_iterable).compute()
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
result = stage3.compute()

# or:
result = Consumer(1000)(Processor(10.0)(Producer(0.5)(input_iterable))).compute()
```

# See also
- [ArrayQueues](https://github.com/portugueslab/arrayqueues) 
- [joblib.Parallel](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
- [Déjà Q](https://en.wikipedia.org/wiki/Deja_Q)
