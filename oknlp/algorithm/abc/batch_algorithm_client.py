from typing import Any, List
from .algorithm import Algorithm
from multiprocessing.connection import Client
import multiprocessing as mp
import threading
import time
import weakref
import logging
from ...utils.process_io import split_text_list, merge_result

logger = logging.getLogger("oknlp")

def _courier(algorithm_weakref, client):
    while True:
        try:
            result_dict = client.recv()
        except EOFError:
            break
        except OSError:
            # Server stoped
            break
        
        thread_id, result, exc = result_dict['id'], result_dict['result'], result_dict["exception"]

        self = algorithm_weakref()
        if self is None:
            # algorithm class deleted
            break

        with self._result_dict_lock:
            self._result_dict[thread_id]['result'] = result
            self._result_dict[thread_id]["exception"] = exc
            self._result_dict[thread_id]['event'].set()
        del self

class BatchAlgorithmClient(Algorithm):
    def __init__(self, address, family, server_name, split_sent):
        self.__address = address
        self.__family = family
        self.__server_name = server_name
        self.__closed = True
        self.split_sent = split_sent
        self.__reinit()
    
    def __getstate__(self):
        return (self.__address, self.__family, self.__closed, self.__server_name)

    def __setstate__(self, state):
        (self.__address, self.__family, self.__closed, self.__server_name) = state
        if not self.__closed:
            self.__reinit()

    def __reinit(self):
        logger.info("[Process %d %s]: Client reinit called", mp.current_process().pid, self.__server_name)
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
        
        self.__closed = False

        logger.info("[Process %d %s]: Client connected", mp.current_process().pid, self.__server_name)

        self.client_lock = threading.Lock()  # write lock
        self.courier_thread = threading.Thread(target=_courier, args=(weakref.ref(self), self.client), daemon=True)
        self.courier_thread.start()
    
    def close(self):
        if self.__closed:
            return
        try:
            self.client.send({"op": 2})
        except BrokenPipeError:
            # server already exited
            pass
        self.client.close()

        logger.info("[Process %d %s]: Client disconnected", mp.current_process().pid, self.__server_name)
        self.__closed = True

    def __del__(self):
        logger.info("[Process %d %s]: __del__ called", mp.current_process().pid, self.__server_name)
        self.close()
    
    def __call__(self, sents : List[Any], max_length = 128):
        if self.split_sent:
            print('++++++++++++++++++++++++++++')
            sents, is_end = split_text_list(sents, max_length)
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
        if self.split_sent:
            return merge_result(result, is_end)
        else:
            return result
