import gc, sys, traceback, time, datetime
import logging
logging.basicConfig(level=logging.INFO)

import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import threading
import concurrent.futures

import numpy as np
import cloudpickle

ctx = mp.get_context("spawn")

def passthrough(item):
    return item

def printer(*args, **kwargs):
    print(args, kwargs)

class lazymap:
    ''' Returns an Iterable, functionally related to map (except that outputs are calculated by a pool of processes).
    
    Args:
        fcn (callable): function that is being mapped. Signature: fcn(item, **kwargs)
        it (iterable): iterable that maps over the function, providing items as arguments
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
        **kwargs: optional, being passed to fcn

    Returns: 
        (iterable): an iterable that returns the results of fcn(item) for each item in it
    '''

    def __init__(self, fcn, it, n_workers=1, buffer_bytes=10e6, **kwargs):
        if n_workers > 1 and not isinstance(it, lazymap):
            it = lazymap(None, it, n_workers=1, buffer_bytes=buffer_bytes) # to make sure that the input iterable is thread-safe  
        self._it = it
        if isinstance(self._it, lazymap):
            it = it._out_queue
        self._out_queue = DejaQueue(buffer_bytes)
        self._in_sem  = [ctx.Semaphore(0) for _ in range(n_workers)]
        self._out_sem = [ctx.Semaphore(0) for _ in range(n_workers)]
        self._in_sem[0].release()
        self._out_sem[0].release()
        self._n_workers = n_workers
        worker_args = [cloudpickle.dumps(fcn), it, self._out_queue, n_workers, self._in_sem, self._out_sem]
        self._workers = [ctx.Process(target=self._worker_fcn, args=(pid, *worker_args), kwargs=kwargs) for pid in range(n_workers)]
        [w.start() for w in self._workers]
        self.generator = self._lazymap_generator(self._out_queue)

    def __len__(self):
        return len(self._it)

    def __iter__(self):
        """ Returns an iterator over the results of fcn(item) for each item in it """
        return self.generator
    
    @staticmethod
    def _worker_fcn(pid, pickled_fcn, it, _out_queue, _n_workers, in_sem, out_sem, **kwargs):
        import cloudpickle, traceback
        fcn = cloudpickle.loads(pickled_fcn) if pickled_fcn is not None else None
        nxt = (pid + 1) % _n_workers
        iterator = iter(it)

        while True:
            in_sem[pid].acquire()
            try:
                item = next(iterator)
            except StopIteration:
                in_sem[nxt].release()
                break
            in_sem[nxt].release()
            try:
                res = item if fcn is None else fcn(item, **kwargs)
            except Exception as e:
                tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                res = {"type":"exception","exc_type":str(type(e)), "exc_msg":str(e), "traceback":tb}
            out_sem[pid].acquire()
            _out_queue.put(res)
            out_sem[nxt].release()

        out_sem[pid].acquire()
        _out_queue._signal_stop()
        out_sem[nxt].release()

    @staticmethod
    def _lazymap_generator(_out_queue):
        for item in _out_queue:
            if isinstance(item, dict) and item.get("type") == "exception":
                raise RuntimeError(f"Exception in worker: {item['exc_type']}: {item['exc_msg']}\nTraceback:\n{item['traceback']}")
            yield item
        _out_queue._signal_stop() # in case multiple workers, make sure that all are stopped

    def close(self):
        ''' Sends a termination signal to the workers, waits for them to finish, and deletes the queues.
        ''' 
        self._out_queue.close()
        while not self._out_queue.done:
            time.sleep(0.01)
        for w in self._workers:
            w.join(timeout=.2)
        self._out_queue = None

    def compute(self, progress=True, ndarray=True, tqdm_args={}):
        ''' Computes the results of the lazymap and returns them as a list or ndarray

        Args:
            progress (bool): whether to show a tqdm progress bar (default: True)
            ndarray (bool): whether to try to return the results as a numpy array (default: True)
            tqdm_args: 

        Returns:
            (list): a list of the results of fcn(item) for each item in it
        '''
        iterable = self
        if progress:
            from tqdm.auto import tqdm
            iterable = tqdm(iterable, **tqdm_args)
        out = list(iterable)
        if ndarray:
            try:
                return np.array(out)
            except:
                pass
        else:
            return out

    def submit(self):
        ''' Submits the computation to a background thread and returns a Future object.

        Returns:
            (Future): a Future object that can be used to fetch the result later
        '''
        future = concurrent.futures.Future()
        def run():
            try:
                result = self.compute(tqdm_args=dict(disable=True))
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return future
    

def Parallel(n_workers=1, buffer_bytes=10e6):
    ''' A wrapper to make a class or callablle a parallel worker. Can be used as a decorator.

    Args:
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)

    Returns:

    '''
    def decorator(cls):
        if isinstance(cls, type):
            return WorkerWrapper(cls, n_workers, buffer_bytes)
        elif callable(cls):
            def wrapped(iterable, **kwargs):
                return lazymap(cls, iterable, n_workers, buffer_bytes, **kwargs)
            return wrapped
        else: 
            raise ValueError(f'Invalid type {type(cls)}')
    return decorator

class WorkerWrapper:
    ''' A helper class used by the Parallel decorator to wrap a class to make it a parallel worker.

    Args:
        cls (class): the class to be wrapped
        n_workers (int): number of workers (default: 1)
        buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
    '''
    def __init__(self, cls, n_workers=1, buffer_bytes=10e6):
        self._instance = None
        self._cls = cls
        self._n_workers = n_workers
        self._buffer_bytes = int(buffer_bytes)
    def __call__(self, *init_args, **init_kwargs):
        def mapper(iterable, **map_kwargs):
            def worker(arg, **kwargs): 
                if self._instance is None:
                    self._instance = self._cls(*init_args, **init_kwargs)
                return self._instance(arg, **kwargs)
            return lazymap(worker, iterable, self._n_workers, self._buffer_bytes, **map_kwargs)
        return mapper


class Startable:
    """ Wrap an iterable so iteration blocks until .start() is called. Works across processes.

    Args:
        source (iterable or callable): an iterable or a zero-arg factory function that produces
        emit_one (bool): if True, iteration will pause after each item until .start() is called again
        ctx (str): multiprocessing context (default: "spawn")
    """
    def __init__(self, source, emit_one=False):
        self._start = ctx.Event()
        self._done  = ctx.Event()
        self._source = source      # iterable OR zero-arg factory
        self._it = None
        self._emit_one = ctx.Value("b", emit_one)

    def start(self):
        self._start.set()
        return self
    
    def pause(self):
        self._start.clear()
        return self

    def stop(self):
        self._done.set()
        self._start.set()          # release any waiters
        return self

    def __iter__(self):
        return self

    def __next__(self):
        self._start.wait()
        if self._done.is_set():
            raise StopIteration
        if self._it is None:
            src = self._source() if callable(self._source) else self._source
            self._it = iter(src)   # created in the consumer process
        if self._emit_one.value:
            self._start.clear()
        return next(self._it)
    
class Ticker(Startable):
    """ An iterable that yields the current datetime when started.
    """
    def __init__(self, emit_one=False):
        super().__init__(iter(datetime.datetime.now, object()), emit_one=emit_one)


# class OrderedStage:
#     ''' A a processing stage, consisting of one or more workers, that processes items in order.
    
#     Args:
#         fcn (callable): function that is being called on each item. Signature: fcn(item, **kwargs)
#         n_workers (int): number of workers (default: 1)
#         buffer_bytes (int): size of the queue buffer (default: 10e6 bytes)
#         **kwargs: optional, being passed to fcn
#     '''
#     def __init__(self, fcn, n_workers=1, buffer_bytes=10e6, in_queue = None, start=True, **kwargs):
#         self._in_queue = in_queue or LegacyDejaQueue(buffer_bytes)
#         self._out_queue = LegacyDejaQueue(buffer_bytes)
#         self._k = mp.Value("l", 0)
#         self._k_changed = mp.Condition()
#         self._n_workers = n_workers
#         self.fcn = fcn
#         self.kwargs = kwargs
#         if start:
#             self.start()

#     def _worker_fcn(self, pid, fcn, **kwargs):
#         np.random.seed(pid)
#         while not self._in_queue.done:
#             item = self._in_queue.get()
#             if item is None: break
#             res = fcn(item, **kwargs)
#             with self._k_changed:
#                 self._k_changed.wait_for(lambda: self._k.value % self._n_workers == pid)
#                 self._out_queue.put(res)
#                 self._k.value += 1
#                 self._k_changed.notify_all()

#     def put(self, item):
#         ''' Puts an item in the queue to be processed by the workers
        
#         Args:
#             item: the item to be processed
#         '''
#         self._in_queue.put(item)

#     def get(self):
#         ''' Gets the next result from the workers'''
#         return self._out_queue.get()
    
#     def start(self):
#         ''' Starts the workers'''
#         self._workers = [mp.Process(target=self._worker_fcn, args=(pid, self.fcn), kwargs=self.kwargs) for pid in range(self._n_workers)]
#         [w.start() for w in self._workers]

#     def close(self):
#         ''' Sends a termination signal to the workers, waits for them to finish, and deletes the queues.
#         ''' 
#         [self._in_queue.put(None) for _ in self._workers]
#         [w.join() for w in self._workers]
#         self._out_queue.close()
#         self._in_queue.close()

#     def __del__(self):
#         self.close()



# class FutureResult:
#     """ A class that represents a future result of a computation in a parallel worker.
#     """
#     def __init__(self, actor, handle):
#         self._actor = actor
#         self._handle = handle

#     def get(self):
#         """ Fetches the result.
#         """
#         if not hasattr(self, '_result'):
#             if self._handle not in self._actor._result_store:
#                 response = self._actor._out_queue.get()
#                 if response['type'] == 'delayed_result':
#                     self._actor._result_store[response['handle']] = response['value']
#                     assert self._handle == response['handle'], "Unexpected handle"
#                 elif response['type'] == 'exception':
#                     self._raise_remote_exception(response['exception'])
#                 else: 
#                     raise ValueError(f"Unexpected message: {response}")
#             self._result = self._actor._result_store[self._handle]
#             del self._actor._result_store[self._handle]
#         return self._result

#     def __del__(self):
#         #making sure that the result is fetched from the store before the object is deleted
#         self.get()


# class Actor:
#     """ A class that wraps an object in another process, allowing you to call methods 
#     and set/get properties remotely via queues. Calling `some_method.delayed(...)` 
#     returns a FutureResult object.
#     """

#     def __init__(self, cls, *args, **kwargs):
#         """
#         Initialize an Actor object.

#         Args:
#             cls (type): The class of the object to be instantiated in the subprocess.
#             *args: Positional arguments to be passed to the constructor of the class.
#             **kwargs: Keyword arguments to be passed to the constructor of the class.
#         """
#         self._cls = cls
#         self._in_queue = DejaQueue()
#         self._out_queue = DejaQueue()
#         self._process = mp.Process(target=self._run, args=(cls, self._in_queue, self._out_queue, args, kwargs))
#         self._process.start()
#         self._result_store = {}
#         response = self._out_queue.get()
#         if response['type'] == 'exception':
#             self._raise_remote_exception(response['exception'])
#         elif not (response['type'] == 'OK'):
#             raise ValueError(f"Invalid response type: {response['type']}")

#     @staticmethod
#     def _run(cls, in_queue, out_queue, args, kwargs):
#         """
#         The method that runs in the subprocess. It instantiates the object of the
#         given class and processes messages from the in_queue.

#         Args:
#             cls (type): The class of the object to be instantiated.
#             in_queue (DejaQueue): Queue for receiving messages from the main process.
#             out_queue (DejaQueue): Queue for sending responses to the main process.
#             args (tuple): Positional arguments to pass to the constructor of the class.
#             kwargs (dict): Keyword arguments to pass to the constructor of the class.
#         """
#         def serialize_exception(e):
#             """
#             Helper function to serialize exception details.

#             Args:
#                 e (Exception): The exception to serialize.

#             Returns:
#                 dict: A dictionary containing exception type, message, and traceback.
#             """
#             exc_type, exc_value, exc_traceback = sys.exc_info()
#             return {'type': exc_type.__name__, 'message': str(e), '\nTraceback': ''.join(traceback.format_tb(exc_traceback))}

#         try:
#             obj = cls(*args, **kwargs)
#             out_queue.put(dict(type='OK'))
#         except Exception as e:
#             out_queue.put(dict(type='exception', exception=serialize_exception(e)))
#             return
        
#         while True:
#             msg = in_queue.get()
#             if msg is None: 
#                 del obj
#                 gc.collect()
#                 break
#             try:
#                 if msg['type'] == 'method':
#                     method = getattr(obj, msg['method'])
#                     out = method(*msg['args'], **msg['kwargs'])
#                     out_queue.put(dict(type='result', value=out))
#                 elif msg['type'] == 'delayed':
#                     method = getattr(obj, msg['method'])
#                     handle = f'handle_{np.random.randint(0, 2**63)}'
#                     out_queue.put(dict(type='delayed_handle', value=handle))
#                     result = method(*msg['args'], **msg['kwargs'])
#                     out_queue.put(dict(type='delayed_result', value=result, handle=handle))
#                 elif msg['type'] == 'get':
#                     out = getattr(obj, msg['attr'])
#                     out_queue.put(dict(type='result', value=out))
#                 elif msg['type'] == 'set':
#                     setattr(obj, msg['attr'], msg['value'])
#                     out_queue.put(dict(type='OK'))
#                 else:
#                     out_queue.put(dict(type='exception', exception={'type': 'ValueError', 'message': f"Invalid message type: {msg['type']}"}))
#             except Exception as e:
#                 out_queue.put(dict(type='exception', exception=serialize_exception(e)))
    
#     def _txrx(self, msg):
#         """ Sends a message to the subprocess and receives a response.
#         """
#         #assert self._in_queue.empty(), "communication queue is not empty"

#         self._in_queue.put(msg)
#         response = self._out_queue.get()
#         if response['type'] == 'delayed_result':
#             self._result_store[response['handle']] = response['value']
#             response = self._out_queue.get()

#         if response['type'] == 'exception':
#             self._raise_remote_exception(response['exception'])
#         elif response['type'] == 'OK':
#             return None
#         elif response['type'] == 'result':
#             return response['value']
#         elif response['type'] == 'delayed_handle':
#             return response['value']
#         else:
#             raise ValueError(f"Invalid response type: {response['type']}")
    
#     def _raise_remote_exception(self, exc_info):
#         """
#         Raises an exception that was caught in the subprocess and passed to the
#         main process.

#         Args:
#             exc_info (dict): Serialized exception details from the subprocess.

#         Raises:
#             Exception: A dynamically created exception based on the subprocess error.
#         """
#         exc_type = type(exc_info['message'], (Exception,), {})
#         raise exc_type(f"{exc_info['type']}: {exc_info['message']}\nTraceback:\n{exc_info['traceback']}")

#     def __getattribute__(self, item):
#         """
#         Retrieves an attribute of the Actor object.

#         Args:
#             item (str): The name of the attribute to retrieve.

#         Returns:
#             Any: The value of the requested attribute.
#         """
#         return super().__getattribute__(item)

#     def __getattr__(self, attr):
#         """
#         Handles getting attributes and methods from the wrapped object in the
#         subprocess. It returns a proxy for method calls or the attribute value.

#         Args:
#             attr (str): The attribute or method name.

#         Returns:
#             Any: A method proxy if the attribute is callable, or the attribute value.

#         Raises:
#             Exception: If there is an error in the subprocess.
#         """        
#         def method_proxy(*args, **kwargs):
#             """ Proxy function that sends method calls to the subprocess.
#             """
#             result = self._txrx(dict(type='method', method=attr, args=args, kwargs=kwargs))
#             return result
        
#         def delayed_proxy(*args, **kwargs):
#             """ Proxy function that sends method calls to the subprocess.
#             """
#             handle = self._txrx(dict(type='delayed', method=attr, args=args, kwargs=kwargs))
#             fut = FutureResult(self, handle)
#             return fut
        
#         method_proxy.delayed = delayed_proxy

#         result = self._txrx(dict(type='get', attr=attr))
#         if callable(result):
#             return method_proxy  # Return the proxy to call the method remotely
#         return result

#     def __setattr__(self, attr, value):
#         """
#         Sets an attribute of the Actor or the wrapped object in the subprocess.

#         Args:
#             attr (str): The name of the attribute.
#             value (Any): The value to set.

#         Raises:
#             AssertionError: If the communication queue is not empty.
#         """
#         if attr.startswith('_'):
#             super().__setattr__(attr, value)
#         else:
#             self._txrx(dict(type='set', attr=attr, value=value))

#     def __call__(self, *args, **kwargs):
#         """
#         Calls the wrapped object's __call__ method.

#         Args:
#             *args: Positional arguments for the method.
#             **kwargs: Keyword arguments for the method.

#         Returns:
#             Any: The result of the __call__ method.
#         """
#         #self._in_queue.put(dict(type='method', method='__call__', args=args, kwargs=kwargs))
#         #return self._out_queue.get()
#         return self._txrx(dict(type='method', method='__call__', args=args, kwargs=kwargs))

#     def __del__(self):
#         """
#         Cleans up the Actor object by terminating the subprocess and joining it.
#         """
#         self._in_queue.put(None)
#         self._process.join()

#     def __dir__(self):
#         """
#         Returns the directory of attributes and methods of the wrapped object.

#         Returns:
#             list: A list of attributes and methods.
#         """
#         # self._in_queue.put(dict(type='method', method='__dir__', args=(), kwargs={}))
#         # obj_attrs = self._out_queue.get() or []
#         obj_attrs = self._txrx(dict(type='method', method='__dir__', args=[], kwargs={})) or []
#         if isinstance(obj_attrs, dict) and obj_attrs['type'] == 'exception':
#             self._raise_remote_exception(obj_attrs['exception'])
#         return obj_attrs

#     def __repr__(self):
#         """
#         Returns a string representation of the Actor, including the wrapped object's
#         representation.

#         Returns:
#             str: A string representing the Actor and the wrapped object.
#         """
#         # self._in_queue.put(dict(type='method', method='__repr__', args=[], kwargs={}))
#         # result = self._out_queue.get()
#         result = self._txrx(dict(type='method', method='__repr__', args=[], kwargs={}))
#         if isinstance(result, dict) and result['type'] == 'exception':
#             return "<Actor (error fetching repr)>"
#         return f"<Actor wrapping: {result}>"
