from queue import Empty
import logging
import multiprocessing as mp
from multiprocessing.connection import Listener, wait
import threading
import queue
import sys, signal

logger = logging.getLogger("oknlp")

class SingleQuery:
    def __init__(self, serial_idx, idx, data, exception=None):
        self.serial_idx = serial_idx
        self.idx = idx
        self.data = data
        self.exception = exception

def handle_sigterm(text):
    def wrapper(*args):
        logger.info("SIGTERM: %s", text)
        sys.exit(0)
    return wrapper
    
class AlgorithmListener:
    def __init__(self, q_input: mp.Queue, q_output: mp.Queue, server_stop_event, address=None, family=None) -> None:
        self.listener = Listener(address=address, family=family)
        self.conn_list = []
        self.q_input = q_input
        self.q_output = q_output
        self.server_lock = threading.Lock()
        self.serial_id = 0
        self.request_map = {}
        self.req_lock = threading.Lock()
        self.first_client_init = threading.Event()
        self.server_stop_event = server_stop_event

    def _thread_listener(self):
        while True:
            conn = self.listener.accept()
            logger.info("Client connected")
            conn.send({
                "op": 0,
                "msg": "hello"
            })
            with self.server_lock:
                if len(self.conn_list) == 0:
                    self.first_client_init.set()
                self.conn_list.append(conn)

    def _thread_scatter(self):
        self.first_client_init.wait()
        while True:
            with self.server_lock:
                self.conn_list = list(filter(lambda x: not x.closed, self.conn_list))
                if len(self.conn_list) == 0:
                    # all clients disconnected
                    self.server_stop_event.set()
                    break

            for conn in wait(self.conn_list, 1):
                try:
                    request = conn.recv()
                except EOFError:
                    if not conn.closed:
                        conn.close()
                    logger.info("Client disconnected")
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

    def join(self):
        signal.signal(signal.SIGTERM, handle_sigterm("exit algorithm listener"))

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


class BatchAlgorithmServer:
    def __init__(self, algorithm,  batch_size, num_preprocess, num_postprocess, max_queue_size, multiprocessing_context) -> None:
        if num_preprocess is None:
            num_preprocess = min(mp.cpu_count(), 4)
        if num_postprocess is None:
            num_postprocess = min(mp.cpu_count(), 4)
        if multiprocessing_context is None and "fork" in mp.get_all_start_methods():
            multiprocessing_context = "fork"
        self.algorithm = algorithm
        multiprocessing = mp.get_context(multiprocessing_context)
        raw_queue = multiprocessing.Queue(max_queue_size)  # raw
        pre_queue = multiprocessing.Queue(max_queue_size)  # after preprocess
        infer_queue = multiprocessing.Queue(max_queue_size)  # after infer
        post_queue = multiprocessing.Queue(max_queue_size)  # after postprocess
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
        __evt_server_stop = mp.Event()

        p_preprocess = [
            multiprocessing.Process (
                name = "%s-preprocess-%d" % (algorithm.__class__.__name__, i),
                target = self._preprocess, 
                args = (raw_queue, pre_queue),
                daemon = True
            ) for i in range(num_preprocess)
        ]
        p_infer = multiprocessing.Process (
            name = "%s-inference" % algorithm.__class__.__name__,
            target = self._infer, 
            args = (pre_queue, infer_queue, batch_size),
            daemon = True
        )
        p_postprocess = [
            multiprocessing.Process (
                name = "%s-postprocess-%d" % (algorithm.__class__.__name__, i),
                target = self._postprocess, 
                args = (infer_queue, post_queue),
                daemon = True
            ) for i in range(num_postprocess)
        ]
        p_listener = multiprocessing.Process (
            name = "%s-server" % algorithm.__class__.__name__,
            target = self._listener, 
            args = (raw_queue, post_queue, __evt_server_stop, self.__address, self.__family, __evt_server_start), 
            daemon = True
        )

        for p in p_preprocess:
            p.start()
        p_infer.start()
        for p in p_postprocess:
            p.start()
        p_listener.start()

        p_wait = threading.Thread(name="%s-wait-stop" % algorithm.__class__.__name__, target=self._wait_stop_thread, args=(__evt_server_stop,), daemon=True)
        p_wait.start()

        # wait until listener started
        __evt_server_start.wait()
        logger.info("Algorithm server %s started", algorithm.__class__.__name__)

        self.p_preprocess = p_preprocess
        self.p_infer = p_infer
        self.p_postprocess = p_postprocess
        self.p_listener = p_listener
        self.__closed = False
    
    def close(self):
        if self.__closed:
            return
        logger.info("Algorithm server %s stoped", self.algorithm.__class__.__name__)
        all_sub_processes = [self.p_infer, self.p_listener] + self.p_preprocess + self.p_postprocess
        for process in all_sub_processes:
            process.terminate()
        for process in all_sub_processes:
            if process.is_alive():
                process.join(1)
                process.kill()
        self.__closed = True
        

    def _wait_stop_thread(self, evt):
        evt.wait()
        self.close()
    
    def __del__(self):
        self.close()
    
    def _listener(self, q_input: mp.Queue, q_output: mp.Queue, __evt_server_stop, address, family, __evt_server_start):
        server = AlgorithmListener(q_input, q_output, __evt_server_stop, address, family)
        server.start()
        __evt_server_start.set()
        server.join()
        
    def _preprocess(self, from_queue: mp.Queue, to_queue: mp.Queue):
        signal.signal(signal.SIGTERM, handle_sigterm("exit preprocess"))
        self.algorithm.init_preprocess()
        while True:
            try:
                query = from_queue.get()
            except KeyboardInterrupt:
                break
            try:
                query.data = self.algorithm.preprocess(query.data)
            except InterruptedError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                query.exception = e
                query.data = None
            to_queue.put(query)

    def _infer(self, from_queue: mp.Queue, to_queue: mp.Queue, batch_size):
        signal.signal(signal.SIGTERM, handle_sigterm("exit inference"))
        self.algorithm.init_inference()

        batch_queue = queue.Queue(1)
        builder = BatchBuilder(from_queue, batch_queue, to_queue, batch_size, self.algorithm.pack_batch)
        t_batch = threading.Thread(target = builder.main, daemon=True)
        t_batch.start()
        
        while True:
            try:
                (batch_info, batch_data) = batch_queue.get()
            except KeyboardInterrupt:
                break
            try:
                builder.inference = True
                batch_data = self.algorithm.inference(batch_data)
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
        signal.signal(signal.SIGTERM, handle_sigterm("exit postprocess"))
        self.algorithm.init_postprocess()
        while True:
            try:
                query = from_queue.get()
            except KeyboardInterrupt:
                break
            if query.exception is not None:
                to_queue.put(query)
                continue
            try:
                query.data = self.algorithm.postprocess(query.data)
            except KeyboardInterrupt:
                break
            except InterruptedError:
                break
            except Exception as e:
                query.data = None
                query.exception = e
            to_queue.put(query)
    
    @property
    def address(self):
        return self.__address, self.__family
        