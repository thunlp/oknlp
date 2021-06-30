from .algorithm import Algorithm
from .batch_algorithm_server import BatchAlgorithmServer
from .batch_algorithm_client import BatchAlgorithmClient
import copyreg

class BatchAlgorithm(Algorithm):
    def __new__(cls, *args, **kwargs):
        algorithm = super().__new__(cls)
        algorithm.__init__(*args, **kwargs)
        server = BatchAlgorithmServer(algorithm, **algorithm.settings)
        client = BatchAlgorithmClient(*server.address) 
        return client

    def __init__(self, batch_size=1, num_preprocess=None, num_postprocess=None, max_queue_size=1024, multiprocessing_context = None):
        self.settings = {
            "batch_size": batch_size,
            "num_preprocess": num_preprocess,
            "num_postprocess": num_postprocess,
            "max_queue_size": max_queue_size,
            "multiprocessing_context": multiprocessing_context,
        }
    
    def __reduce_ex__(self, proto):
        # using __reduce__ to call object.__new__ instead of BatchAlgorithm.__new__
        return copyreg._reconstructor, (self.__class__, object, None), self.__dict__
    
    def pack_batch(self, batch):
        return batch
    
    def init_preprocess(self):
        pass

    def init_inference(self):
        pass

    def init_postprocess(self):
        pass

    def preprocess(self, x):
        return x

    def inference(self, batch):
        return batch

    def postprocess(self, x):
        return x
