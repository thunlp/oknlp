import unittest
import os, shutil

HOME = os.environ["HOME"]
DATA_DIR = os.path.abspath("./testdir")

class TestDataManager(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs(DATA_DIR)
    
    def tearDown(self) -> None:
        shutil.rmtree(DATA_DIR)
    
    def test_load(self):
        from ink.config import config
        from ink.data import load

        old = config.path
        config.path = [ DATA_DIR ]
        path = load("test")
        config.path = old

        self.assertTrue(path.startswith(DATA_DIR))
        
        with self.assertRaises(ValueError):
            load("unknown")
    


    
