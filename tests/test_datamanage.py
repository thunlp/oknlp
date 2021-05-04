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
        from oknlp import load, config

        old = config.path
        config.path = [ DATA_DIR ]
        path = load("test")
        config.path = old

        self.assertTrue(path.startswith(DATA_DIR))
        
        with self.assertRaises(ValueError):
            load("unknown")
    
    def test_loadpath(self):
        from oknlp import load, config
        old = config.path

        config.path = [os.path.join(DATA_DIR, "path1")]
        v = load("test")
        self.assertEqual(open(os.path.join(v, "test.txt"), "r").read(), "test")

        config.path = [os.path.join(DATA_DIR, "path2"), "asdasd", "/bbbb", os.path.join(DATA_DIR, "path1"), "?????"]
        vv = load("test")
        self.assertEqual( vv, v )
        self.assertEqual(open(os.path.join(vv, "test.txt"), "r").read(), "test")
        self.assertTrue(not os.path.exists(os.path.join(DATA_DIR, "sources", "test")) )

        config.path = old

    


    
