import multiprocessing as mp
from multiprocessing.connection import Client, Listener, wait
import queue
from queue import Empty
import threading
import sys
import time
import warnings

class SingleQuery:
    def __init__(self, serial_idx, idx, data, exception=None):
        self.serial_idx = serial_idx
        self.idx = idx
        self.data = data
        self.exception = exception

class AlgorithmServer:
    def __init__(self, q_input: mp.Queue, q_output: mp.Queue, address=None, family=None) -> None:
        self.listener = Listener(address=address, family=family)
        self.conn_list = []
        self.q_input = q_input
        self.q_output = q_output
        self.server_lock = threading.Lock()
        self.serial_id = 0
        self.request_map = {}
        self.req_lock = threading.Lock()
        self.first_client_init = threading.Event()

    def _thread_listener(self):
        while True:
            conn = self.listener.accept()
            conn.send({
                "op": 0,
                "msg": "hello"
            })
            with self.server_lock:
                if len(self.conn_list) == 0:
                    self.first_client_init.set()
                self.conn_list.append(conn)

    def _thread_scatter(self):
        while True:
            with self.server_lock:
                self.conn_list = list(filter(lambda x: not x.closed, self.conn_list))
                if len(self.conn_list) == 0:
                    self.first_client_init.clear()
            self.first_client_init.wait()

            for conn in wait(self.conn_list, 1):
                try:
                    request = conn.recv()
                except EOFError:
                    continue
                if request["op"] == 1:
                    with self.req_lock:
                        self.request_map[self.serial_id] = {
                            "conn": conn,
                            "total_size": len(request["data"]),
                            "result": [],
                            "exception": None,
                            "id": request["id"]
                        }
                        serial_idx = self.serial_id
                        self.serial_id += 1

                    # put in queue
                    for idx, it in enumerate(request["data"]):
                        self.q_input.put(SingleQuery(serial_idx, idx, it))

    def _thread_gather(self):
        while True:
            query = self.q_output.get()
            with self.req_lock:
                request = self.request_map[query.serial_idx]
                request["result"].append((query.idx, query.data))
                if query.exception is not None:
                    request["exception"] = query.exception
                
                if len(request["result"]) == request["total_size"]:
                    # send back here
                    # sort by index
                    sorted_results = sorted(request["result"], key=lambda x: x[0])

                    # remove index
                    list_results = list(map(lambda x: x[1], sorted_results))
                    if not request["conn"].closed:
                        request["conn"].send({"result": list_results, "id": request["id"], "exception": request["exception"]})
                # else do nothing

    def start(self):
        self.t_listener = threading.Thread(target=self._thread_listener, daemon=True)
        self.t_scatter = threading.Thread(target=self._thread_scatter, daemon=True)
        self.t_gather = threading.Thread(target=self._thread_gather, daemon=True)

        self.t_listener.start()
        self.t_scatter.start()
        self.t_gather.start()

        self.t_listener.join()

class BatchBuilder:
    def __init__(self, from_queue : queue.Queue, to_queue : queue.Queue, bypass_queue : queue.Queue, batch_size : int, pack_func = None):
        self.from_queue = from_queue
        self.to_queue = to_queue
        self.bypass_queue = bypass_queue
        self.pack_func = pack_func
        self.inference = False
        self.batch_size = batch_size
    
    def main(self):
        """一个线程负责组装batch
        """
        while True:
            batch_info = []
            batch_data = []

            try:
                query = self.from_queue.get()
                while True:
                    # append query
                    if query is None:
                        # query is None when queue is empty and model is infering
                        pass
                    else:
                        if query.exception is not None:
                            self.bypass_queue.put(query)
                        else:
                            batch_info.append((query.serial_idx, query.idx))
                            batch_data.append(query.data)
                    
                    if len(batch_data) >= self.batch_size:
                        # 1. == batch_size
                        break
                    
                    try:
                        # try to get more
                        query = self.from_queue.get_nowait()
                    except Empty:
                        # 2. Queue is empty
                        if self.inference:
                            query = None
                            continue
                        else:
                            break
                if len(batch_data) == 0: # happends when the first element of this batch has an exception
                    continue

                if self.pack_func is not None:
                    batch_data = self.pack_func(batch_data)
                self.to_queue.put((batch_info, batch_data))
            except InterruptedError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                for serial_idx, idx in batch_info:
                    self.bypass_queue.put( SingleQuery(serial_idx, idx, None, e) )
            



class BaseAlgorithm:
    '''算法类的基类，派生类需要实现preprocess(self, x)、infer(self, batch)、postprocess(self, x)方法
    '''
    def __init__(self, batch_size=1, num_preprocess=None, num_postprocess=None, max_queue_size=1024, multiprocessing_context = None):
        if num_preprocess is None:
            num_preprocess = min(mp.cpu_count(), 4)
        if num_postprocess is None:
            num_postprocess = min(mp.cpu_count(), 4)
        self.batch_size = batch_size

        if multiprocessing_context is None and "fork" in mp.get_all_start_methods():
            multiprocessing_context = "fork"
        multiprocessing = mp.get_context(multiprocessing_context)

        self.raw_queue = multiprocessing.Queue(max_queue_size)  # raw
        self.pre_queue = multiprocessing.Queue(max_queue_size)  # after preprocess
        self.infer_queue = multiprocessing.Queue(max_queue_size)  # after infer
        self.post_queue = multiprocessing.Queue(max_queue_size)  # after postprocess

        if sys.platform == "win32":
            # using named pipe
            import random
            RANDOM_NAME_LIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOOPQRSTUVWXYZ1234567890"
            self.__address = r"\\.\pipe" + "\\" + "".join(random.choices(RANDOM_NAME_LIST, k=16))
            self.__family = "AF_PIPE"
        else:
            # using unix socket
            import tempfile
            self.__address = tempfile.mktemp()
            self.__family = "AF_UNIX"
        __evt_server_start = mp.Event()
        p_preprocess = [
            multiprocessing.Process (
                name = "%s-preprocess-%d" % (self.__class__.__name__, i),
                target = self._preprocess, 
                args = (self.raw_queue, self.pre_queue), 
                daemon = True
            ) for i in range(num_preprocess)
        ]
        p_infer = multiprocessing.Process (
            name = "%s-inference" % self.__class__.__name__,
            target = self._infer, 
            args = (self.pre_queue, self.infer_queue), 
            daemon=True
        )
        p_postprocess = [
            multiprocessing.Process (
                name = "%s-postprocess-%d" % (self.__class__.__name__, i),
                target = self._postprocess, 
                args = (self.infer_queue, self.post_queue), 
                daemon=True
            ) for i in range(num_postprocess)
        ]
        p_listener = multiprocessing.Process (
            name = "%s-server" % self.__class__.__name__,
            target = self._listener, 
            args = (self.raw_queue, self.post_queue, self.__address, self.__family, __evt_server_start), 
            daemon=True
        )

        for p in p_preprocess:
            p.start()
        p_infer.start()
        for p in p_postprocess:
            p.start()
        p_listener.start()

        # wait until listener started
        __evt_server_start.wait()

        self.p_preprocess = p_preprocess
        self.p_infer = p_infer
        self.p_postprocess = p_postprocess
        self.p_listener = p_listener

        self._reinit_client()

    def _listener(self, q_input: mp.Queue, q_output: mp.Queue, address, family, __evt_server_start):
        server = AlgorithmServer(q_input, q_output, address, family)
        __evt_server_start.set()
        server.start()

    def _reinit_client(self):
        self.__closed = False
        self._result_dict = {}
        self._result_dict_lock = threading.Lock()

        client_ok = False
        for _ in range(3):
            try:
                self.client = Client(self.__address, self.__family)
            except ConnectionRefusedError:
                time.sleep(0.5)
            else:
                client_ok = True
                break
        if not client_ok:
            raise RuntimeError("Failed to init client")
        response = self.client.recv()
        if response["op"] == 0:
            assert response["msg"] == "hello"
        self.client_lock = threading.Lock()  # write lock
        self.courier_thread = threading.Thread(target=self._courier, daemon=True)
        self.courier_thread.start()

    def _courier(self):
        '''持续从client中recv，每得到一个结果，就（获取dict锁后）放入dict，激活event
        '''
        while True:
            try:
                result_dict = self.client.recv()
            except EOFError:
                break

            thread_id, result, exc = result_dict['id'], result_dict['result'], result_dict["exception"]
            with self._result_dict_lock:
                self._result_dict[thread_id]['result'] = result
                self._result_dict[thread_id]["exception"] = exc
                self._result_dict[thread_id]['event'].set()

    def __getstate__(self):
        need_reinit_client = hasattr(self, 'client')
        if not hasattr(self, "config"):
            self.config = {}
        return (self.batch_size, self.__address, self.__family, need_reinit_client, self.config)

    def __setstate__(self, state):
        (self.batch_size, self.__address, self.__family, need_reinit_client, self.config) = state
        if need_reinit_client:
            self._reinit_client()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            warnings.warn("There was an accident when oknlp exited. %s" % e)

    def close(self):
        if self.__closed:
            return
        for p in self.p_preprocess:
            p.terminate()
        self.p_infer.terminate()
        for p in self.p_postprocess:
            p.terminate()
        self.p_listener.terminate()
        self.client.close()
        self.__closed = True

    def __call__(self, sents):
        '''线程根据自己的id，（获取client锁后）用client发数据，等待event，获取结果
        '''
        thread_id = threading.get_ident()
        with self._result_dict_lock:
            if thread_id not in self._result_dict:
                self._result_dict[thread_id] = {'event': threading.Event(), 'result': None}
            event = self._result_dict[thread_id]['event']
            event.clear()
        with self.client_lock:
            self.client.send({
                "op": 1,
                "total_size": len(sents),
                "data": sents,
                "id": thread_id
            })
        event.wait()
        with self._result_dict_lock:
            result = self._result_dict[thread_id]['result']
            del self._result_dict[thread_id]['result']
            exc = self._result_dict[thread_id]['exception']
            del self._result_dict[thread_id]['exception']
            if exc is not None:
                raise exc
        return result

    def _preprocess(self, from_queue: mp.Queue, to_queue: mp.Queue):
        """将数据从from_queue中取出，预处理后放入to_queue
        """
        self.init_preprocess()
        while True:
            try:
                query = from_queue.get()
            except KeyboardInterrupt:
                break
            try:
                query.data = self.preprocess(query.data)
            except InterruptedError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                query.exception = e
                query.data = None
            to_queue.put(query)

    def _infer(self, from_queue: mp.Queue, to_queue: mp.Queue):
        """将数据从from_queue中取出，推理后放入to_queue
        """
        
        self.init_inference()

        batch_queue = queue.Queue(1)
        builder = BatchBuilder(from_queue, batch_queue, to_queue, self.batch_size, self.pack_batch)
        t_batch = threading.Thread(target = builder.main, daemon=True)
        t_batch.start()
        
        while True:
            try:
                (batch_info, batch_data) = batch_queue.get()
            except KeyboardInterrupt:
                break
            try:
                builder.inference = True
                batch_data = self.inference(batch_data)
                builder.inference = False
            except InterruptedError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                for serial_idx, idx in batch_info:
                    to_queue.put(SingleQuery(serial_idx, idx, None, e))
            else:
                for info, data in zip(batch_info, batch_data):
                    serial_idx, idx = info
                    to_queue.put( SingleQuery(serial_idx, idx, data) )

    def _postprocess(self, from_queue: mp.Queue, to_queue: mp.Queue):
        """将数据从from_queue中取出，后处理后放入to_queue
        """
        self.init_postprocess()
        while True:
            try:
                query = from_queue.get()
            except KeyboardInterrupt:
                break
            if query.exception is not None:
                to_queue.put(query)
                continue
            try:
                query.data = self.postprocess(query.data)
            except KeyboardInterrupt:
                break
            except InterruptedError:
                break
            except Exception as e:
                query.data = None
                query.exception = e
            to_queue.put(query)
    
    def pack_batch(self, batch):
        return batch
    
    def init_preprocess(self):
        pass

    def init_inference(self):
        pass

    def init_postprocess(self):
        pass

    def preprocess(self, x):
        """对一个输入进行预处理
        """
        return x

    def inference(self, batch):
        """对一组输入进行推理
        """
        return batch

    def postprocess(self, x):
        """对一个输入进行后处理
        """
        return x
