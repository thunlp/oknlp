import unittest
import os

HOME = os.path.expanduser("~")
DATA_DIR = os.path.abspath("./testdir")
TESTYAML1 = """
path:
    - %s
    - abcde
default_device: tpu
""" % DATA_DIR

TESTYAML2 = """
path:   # path here
    - abcde
source: "12345"
"""

class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        os.makedirs("./testdir")
    
    def tearDown(self) -> None:
        os.removedirs("./testdir")
    
    def test_load_config(self):
        from oknlp.config.config import Config, DEFAULT_CONFIG
        
        with open( os.path.join(HOME, ".oknlp.config.yaml"), "w", encoding="utf-8") as f:
            f.write(TESTYAML1)
        cfg = Config()
        self.assertListEqual(cfg.path, [ DATA_DIR, "abcde" ])
        self.assertEqual( cfg.source, DEFAULT_CONFIG["source"])
        os.unlink( os.path.join(HOME, ".oknlp.config.yaml")  )
    
    def test_create_default_config(self):
        from oknlp.config.config import Config
        cfg = Config()
        self.assertTrue( os.path.exists( os.path.join(HOME, ".oknlp.config.yaml")  ) )
    
    def test_config_overwrite(self):
        from oknlp.config.config import Config, DEFAULT_CONFIG
        with open( os.path.join(HOME, ".oknlp.config.yaml"), "w", encoding="utf-8") as f:
            f.write(TESTYAML1)
        with open( os.path.abspath(".oknlp.config.yaml"), "w", encoding="utf-8") as f:
            f.write(TESTYAML2)
        cfg = Config()

        self.assertListEqual( cfg.path, ["abcde"] )
        self.assertEqual( cfg.source, "12345" )
        os.unlink( os.path.join(HOME, ".oknlp.config.yaml")  )
        os.unlink( os.path.abspath(".oknlp.config.yaml")  )

    