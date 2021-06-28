import unittest
from oknlp.algorithm.BaseAlgorithm import BaseAlgorithm
from multiprocessing.pool import ThreadPool


class MyAlgorithm(BaseAlgorithm):
    def inference(self, batch):
        return batch
    
    def preprocess(self, x):
        return x + 1
    
    def postprocess(self, x):
        return x + 1


class Caller:
    def __init__(self, model) -> None:
        self.model = model
    
    def work(self, x):
        return self.model([x])[0]

class TestExceptions(unittest.TestCase):
    
    def test_exception(self):
        alg = MyAlgorithm(batch_size=16)
        caller = Caller(alg)

        with ThreadPool(processes=32) as pool:
            result = pool.map(caller.work, range(2**10))
        for i, x in enumerate(result):
            self.assertEqual(i + 2, x)
        
        
    