import unittest
import os

HOME = os.environ["HOME"]
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
        
        open( os.path.join(HOME, ".oknlp.config.yaml"), "w" ).write(TESTYAML1)
        cfg = Config()
        self.assertListEqual(cfg.path, [ DATA_DIR, "abcde" ])
        self.assertEqual( cfg.source, DEFAULT_CONFIG["source"])
        self.assertEqual(cfg.default_device, "tpu")
        os.unlink( os.path.join(HOME, ".oknlp.config.yaml")  )
    
    def test_create_default_config(self):
        from oknlp.config.config import Config
        cfg = Config()
        self.assertTrue( os.path.exists( os.path.join(HOME, ".oknlp.config.yaml")  ) )
    
    def test_config_overwrite(self):
        from oknlp.config.config import Config, DEFAULT_CONFIG

        open( os.path.join(HOME, ".oknlp.config.yaml"), "w" ).write(TESTYAML1)
        open( os.path.abspath(".oknlp.config.yaml"), "w" ).write(TESTYAML2)
        cfg = Config()

        self.assertListEqual( cfg.path, ["abcde"] )
        self.assertEqual( cfg.source, "12345" )
        self.assertEqual( cfg.default_device, "tpu" )
        os.unlink( os.path.join(HOME, ".oknlp.config.yaml")  )
        os.unlink( os.path.abspath(".oknlp.config.yaml")  )

    