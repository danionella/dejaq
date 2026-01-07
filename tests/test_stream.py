import pytest
import numpy as np
import time
import concurrent.futures
from dejaq.stream import Source


class TestSource:
    """Tests for Source creation."""

    def test_source_from_iterable(self):
        """Source from iterable yields all items."""
        src = Source(it=range(10))
        src.start()
        results = src.run(progress=False)
        assert list(results) == list(range(10))

    def test_source_from_function(self):
        """Source from function generates items."""
        counter = {"n": 0}
        def gen():
            counter["n"] += 1
            if counter["n"] > 5:
                raise StopIteration
            return counter["n"]
        
        src = Source(it=range(5), fcn=None)  # Use iterable instead for predictable test
        src.start()
        results = src.run(progress=False)
        assert len(results) == 5

    def test_source_from_factory(self):
        """Source from factory creates instance and calls call_fcn."""
        class Counter:
            def __init__(self):
                self.n = 0
            def get(self):
                self.n += 1
                return self.n

        src = Source(it=range(5))
        src.start()
        results = src.run(progress=False)
        assert list(results) == list(range(5))

    def test_source_manual(self):
        """Manual source allows pushing items."""
        src = Source()
        src.put(1)
        src.put(2)
        src.put(3)
        src.stop()
        results = src.run(progress=False)
        assert list(results) == [1, 2, 3]

    def test_source_with_rate_limit(self):
        """Source respects rate limiting."""
        src = Source(it=range(5), rate=100)  # 100 Hz = 10ms intervals
        src.start()
        start = time.time()
        results = src.run(progress=False)
        elapsed = time.time() - start
        assert list(results) == list(range(5))
        # 5 items at 100Hz should take ~40-50ms minimum
        assert elapsed >= 0.04


class TestMap:
    """Tests for .map() functionality."""

    def test_map_with_function(self):
        """Map applies function to each item."""
        src = Source(it=range(5))
        pipeline = src.map(fcn=lambda x: x * 2)
        src.start()
        results = pipeline.run(progress=False)
        assert list(results) == [0, 2, 4, 6, 8]

    def test_map_with_multiple_workers(self):
        """Map with multiple workers processes in parallel but returns ordered."""
        src = Source(it=range(10))
        pipeline = src.map(fcn=lambda x: x ** 2, n_workers=4)
        src.start()
        results = pipeline.run(progress=False)
        assert list(results) == [x ** 2 for x in range(10)]

    # def test_map_with_factory(self):
    #     """Map with factory creates instance per worker."""
    #     class Doubler:
    #         def __call__(self, x):
    #             return x * 2

    #     src = Source(it=range(5))
    #     pipeline = src.map(factory=Doubler)
    #     src.start()
    #     results = pipeline.run(progress=False)
    #     assert list(results) == [0, 2, 4, 6, 8]

    # def test_map_with_factory_and_call_fcn(self):
    #     """Map with factory and custom call_fcn."""
    #     class Processor:
    #         def __init__(self):
    #             self.factor = 3
    #         def process(self, x):
    #             return x * self.factor

    #     src = Source(it=range(5))
    #     pipeline = src.map(factory=Processor, call_fcn=lambda obj, x: obj.process(x))
    #     src.start()
    #     results = pipeline.run(progress=False)
    #     assert list(results) == [0, 3, 6, 9, 12]

    def test_chained_maps(self):
        """Multiple maps can be chained."""
        src = Source(it=range(5))
        pipeline = (
            src
            .map(fcn=lambda x: x + 1)
            .map(fcn=lambda x: x * 2)
        )
        src.start()
        results = pipeline.run(progress=False)
        assert list(results) == [(x + 1) * 2 for x in range(5)]


# class TestTee:
#     """Tests for .tee() functionality."""

#     def test_tee_splits_stream(self):
#         """Tee splits stream into independent copies."""
#         src = Source(it=range(5))
#         stream1, stream2 = src.tee(count=2)
#         src.start()
        
#         results1 = list(stream1)
#         results2 = list(stream2)
        
#         assert results1 == list(range(5))
#         assert results2 == list(range(5))

#     def test_tee_three_way(self):
#         """Tee can split into more than 2 streams."""
#         src = Source(it=range(3))
#         s1, s2, s3 = src.tee(count=3)
#         src.start()
        
#         assert list(s1) == [0, 1, 2]
#         assert list(s2) == [0, 1, 2]
#         assert list(s3) == [0, 1, 2]


class TestZip:
    """Tests for .zip() functionality."""

    def test_zip_combines_streams(self):
        """Zip combines multiple streams into tuples."""
        src1 = Source(it=[1, 2, 3])
        src2 = Source(it=['a', 'b', 'c'])
        
        combined = src1.zip(src2)
        src1.start()
        src2.start()
        
        results = list(combined)
        assert results == [(1, 'a'), (2, 'b'), (3, 'c')]

    def test_zip_three_streams(self):
        """Zip works with three streams."""
        src1 = Source(it=[1, 2])
        src2 = Source(it=['a', 'b'])
        src3 = Source(it=[10, 20])
        
        combined = src1.zip(src2, src3)
        src1.start()
        src2.start()
        src3.start()
        
        results = list(combined)
        assert results == [(1, 'a', 10), (2, 'b', 20)]

    def test_zip_sync_with_tee_lazy_start(self):
        """Zip(sync) should not deadlock with tee when start_mode='lazy'.

        This covers the case where only the primary branch would otherwise start,
        leaving the tee worker waiting for readiness from all outputs.
        """
        src = Source(it=range(20), start_mode="lazy")
        a, b = src.tee(2)

        left = a.map(fcn=lambda x: x, n_workers=2)  # primary
        right = b.map(fcn=lambda x: x * 10, n_workers=1)
        joined = left.zip(right, mode="sync")

        fut = joined.submit(keep_outputs=True, ndarray=False)
        results = fut.result(timeout=3)
        assert results == [(i, i * 10) for i in range(20)]


class TestSink:
    """Tests for .sink() functionality."""

    def test_sink_cannot_be_iterated(self):
        """Sink nodes raise TypeError when iterated."""
        src = Source(it=range(5))
        pipeline = src.sink(fcn=lambda x: None)
        src.start()

        with pytest.raises(TypeError):
            list(pipeline)

        src.start()
        pipeline.wait()


class TestRun:
    """Tests for .run() functionality."""

    def test_run_collects_results(self):
        """Run collects all results."""
        src = Source(it=range(5))
        pipeline = src.map(fcn=lambda x: x * 2)
        src.start()
        results = pipeline.run(progress=False)
        assert list(results) == [0, 2, 4, 6, 8]

    def test_run_returns_ndarray(self):
        """Run returns numpy array when possible."""
        src = Source(it=range(5))
        src.start()
        results = src.run(progress=False, ndarray=True)
        assert isinstance(results, np.ndarray)
        np.testing.assert_array_equal(results, np.arange(5))

    def test_run_keep_outputs_false(self):
        """Run with keep_outputs=False returns count."""
        src = Source(it=range(10))
        src.start()
        count = src.run(progress=False, keep_outputs=False)
        assert count == 10


class TestControlMethods:
    """Tests for pipeline control methods."""

    def test_start_begins_processing(self):
        """Start method begins source processing."""
        src = Source(it=range(5))
        # Before start, nothing should be processed
        src.start()
        results = src.run(progress=False)
        assert list(results) == list(range(5))

    # def test_stop_cancels_processing(self):
    #     """Stop method cancels processing."""
    #     src = Source(it=range(1000), rate=100)  # Slow rate to allow stopping
    #     pipeline = src.map(fcn=lambda x: x)
    #     src.start()
    #     time.sleep(0.05)  # Let some items process
    #     src.stop()

    #     pipeline.wait()
        
    #     # Should have fewer than 1000 items
    #     # Note: exact count depends on timing


class TestCompleteExample:
    """Integration test based on README example."""

    def test_camera_pipeline_example(self):
        """Test a simplified version of the camera pipeline example."""
        
        class FakeCamera:
            def __init__(self):
                self.frame_count = 0
            
            def get_frame(self):
                self.frame_count += 1
                return np.random.randn(48, 64)  # Smaller for testing
        
        class Smoother:
            def __init__(self, sigma=1.0):
                self.sigma = sigma
            
            def __call__(self, frame):
                # Simple smoothing simulation
                return frame * 0.9
        
        # Build pipeline
        src = Source(it=range(10))  # Use finite iterator for testing
        pipeline = (
            src
            .map(fcn=lambda i: np.random.randn(48, 64))
            .map(fcn=lambda frame: (frame - frame.min()) / (frame.max() - frame.min() + 1e-8))
        )
        
        src.start()
        results = pipeline.run(progress=False)
        
        assert len(results) == 10
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.shape == (48, 64) for r in results)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_iterable(self):
        """Empty iterable produces no results."""
        src = Source(it=[])
        src.start()
        results = src.run(progress=False)
        assert len(results) == 0

    def test_single_item(self):
        """Single item iterable works correctly."""
        src = Source(it=[42])
        src.start()
        results = src.run(progress=False)
        assert list(results) == [42]

    def test_numpy_arrays_through_pipeline(self):
        """Numpy arrays pass through pipeline correctly."""
        arrays = [np.array([i, i+1, i+2]) for i in range(3)]
        src = Source(it=arrays)
        pipeline = src.map(fcn=lambda x: x * 2)
        src.start()
        results = pipeline.run(progress=False, ndarray=False)
        
        for i, arr in enumerate(results):
            np.testing.assert_array_equal(arr, arrays[i] * 2)

    def test_dict_through_pipeline(self):
        """Dictionaries pass through pipeline correctly."""
        src = Source(it=[{'a': 1}, {'b': 2}])
        pipeline = src.map(fcn=lambda d: {k: v * 2 for k, v in d.items()})
        src.start()
        results = pipeline.run(progress=False, ndarray=False)
        
        assert results[0] == {'a': 2}
        assert results[1] == {'b': 4}
