
![Python Version](https://img.shields.io/badge/python-3.7+-blue)
[![PyPI - Version](https://img.shields.io/pypi/v/dejaq)](https://pypi.org/project/dejaq/)
[![Conda Version](https://img.shields.io/conda/v/danionella/dejaq)](https://anaconda.org/danionella/dejaq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/danionella/dejaq)

# DéjàQueue

A fast drop-in replacement for `multiprocessing.Queue` taking advantage of a [shared memory](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Array) ring buffer and [pickle protocol 5 with out-of-band data](https://peps.python.org/pep-0574/). As with other queues in Python, data is transmitted via `put` and `get` calls.

DejaQueue supports any type of picklable Python object, including numpy arrays or nested dictionaries with mixed content. It uses locks to support more than one producer and comsumer process.

Auto-generated API documentation: https://danionella.github.io/dejaq

## Usage example
```python
import numpy as np
from multiprocessing import Process
from dqjaq import DejaQueue

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
# See also
- [ArrayQueues](https://github.com/portugueslab/arrayqueues) 