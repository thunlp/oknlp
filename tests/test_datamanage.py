import unittest
import os, shutil

DATA_DIR = os.path.abspath("./testdir")

class TestDataManager(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs(DATA_DIR)
    
    def tearDown(self) -> None:
        shutil.rmtree(DATA_DIR)

    
    def test_load(self):
        from oknlp import load, config

        old = config.path
        config.path = DATA_DIR 
        path = load("test", "")
        config.path = old
        self.assertTrue(path.startswith(DATA_DIR))
        with self.assertRaises(ValueError):
            load("unknown","")
    
    def test_loadpath(self):
        from oknlp import load, config
        old = config.path

        config.path = os.path.join(DATA_DIR, "path1")
        v = load("test", "")
        with open(os.path.join(v, "test.txt"), "r") as f:
            self.assertEqual(f.read(), "test")
        self.assertTrue(not os.path.exists(os.path.join(DATA_DIR, "sources", "test")) )

        config.path = old

    


    
