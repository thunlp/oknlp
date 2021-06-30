import unittest
from oknlp.algorithm.abc import BatchAlgorithm

class MyException1(Exception):
    pass

class MyException2(Exception):
    pass

class MyAlgorithm(BatchAlgorithm):
    def inference(self, batch):
        return batch
    
    def preprocess(self, x):
        if x == 5:
            raise MyException1()
        return x
    
    def postprocess(self, x):
        if x == 6:
            raise MyException2()
        return x

class TestExceptions(unittest.TestCase):
    
    def test_exception(self):
        alg = MyAlgorithm()
        self.assertListEqual(alg([0, 1, 2, 3]), [0, 1, 2, 3])
        with self.assertRaises(MyException1):
            alg([4, 5])
        with self.assertRaises(MyException2):
            alg([6, 7])
        alg.close()


        
    