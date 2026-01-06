import pytest
import numpy as np
import time
import concurrent.futures

from dejaq.parallel import lazymap, Parallel, WorkerWrapper, Startable, Ticker

def test_lazymap_basic():
    f = lambda x: x * 2
    data = list(range(10))
    lm = lazymap(f, data, n_workers=2)
    result = list(lm)
    assert sorted(result) == [x * 2 for x in data]

def test_lazymap_ordered_output():
    def f(x):
        time.sleep(0.01 * (10 - x))  # out of order
        return x
    data = list(range(10))
    lm = lazymap(f, data, n_workers=3)
    result = list(lm)
    assert result == data  # should be ordered

def test_lazymap_compute_ndarray():
    f = lambda x: x + 1
    data = np.arange(5)
    lm = lazymap(f, data, n_workers=2)
    arr = lm.compute(ndarray=True)
    assert isinstance(arr, np.ndarray), "Output is not a numpy array"
    assert np.all(arr == data + 1), f"Array values are incorrect: {arr} != {data + 1}"

def test_lazymap_compute_list():
    f = lambda x: str(x)
    data = [1, 2, 3]
    lm = lazymap(f, data, n_workers=1)
    out = lm.compute(ndarray=False)
    assert out == ['1', '2', '3']

def test_lazymap_submit_future():
    f = lambda x: x ** 2
    data = range(5)
    lm = lazymap(f, data, n_workers=2)
    future = lm.submit()
    result = future.result(timeout=5)
    assert sorted(result) == [x ** 2 for x in data]

def test_lazymap_exception_propagation():
    def f(x):
        if x == 3:
            raise ValueError("bad value")
        return x
    data = range(5)
    lm = lazymap(f, data, n_workers=2)
    with pytest.raises(RuntimeError):
        list(lm)

def test_parallel_function():
    @Parallel(n_workers=3)
    def f(x):
        return x + 10
    data = range(6)
    result = f(data).compute(ndarray=False)
    assert sorted(result) == [x + 10 for x in data]

def test_parallel_class():
    @Parallel(n_workers=2)
    class Adder:
        def __init__(self, offset):
            self.offset = offset
        def __call__(self, x):
            return x + self.offset
    data = [1, 2, 3]
    result = Adder(5)(data).compute(ndarray=False)
    assert sorted(result) == [6, 7, 8]

def test_workerwrapper_direct():
    class Multiplier:
        def __init__(self, factor):
            self.factor = factor
        def __call__(self, x):
            return x * self.factor
    wrapper = WorkerWrapper(Multiplier, n_workers=2)
    data = [2, 3, 4]
    result = wrapper(10)(data).compute(ndarray=False)
    assert sorted(result) == [20, 30, 40]

def test_startable_pause_resume():
    data = [1, 2, 3]
    s = Startable(data)
    it = iter(s)
    s.start()
    assert next(it) == 1
    s.pause()
    s.start()
    assert next(it) == 2
    s.stop()
    with pytest.raises(StopIteration):
        next(it)

def test_ticker_yields_datetime():
    t = Ticker()
    t.start()
    val = next(iter(t))
    import datetime
    assert isinstance(val, datetime.datetime)

# def test_ordered_stage_order():
#     def f(x):
#         time.sleep(0.01 * (5 - x))
#         return x
#     stage = OrderedStage(f, n_workers=2)
#     for i in range(5):
#         stage.put(i)
#     results = [stage.get() for _ in range(5)]
#     assert results == list(range(5))
#     stage.close()

# def test_ordered_stage_close():
#     def f(x): return x
#     stage = OrderedStage(f, n_workers=1)
#     stage.put(1)
#     assert stage.get() == 1
#     stage.close()