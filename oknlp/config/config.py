import os
import yaml


HOME = os.path.expanduser("~")
CONFIG_FILE = ".oknlp.config.yaml"
DEFAULT_CONFIG = {
    "path": [os.path.join(HOME, ".oknlp")],
    "source": "https://data.thunlp.org/ink/",
    "default_device": "cpu"
}


class Config:
    """
    Attributes:
        path: list[str]
        source: str, data source url ending with "/"
        default_device: str
    """

    def __init__(self):
        self.path = DEFAULT_CONFIG["path"]
        self.source = DEFAULT_CONFIG["source"]
        self.default_device = DEFAULT_CONFIG["default_device"]

        self.set_config_from_file(HOME)
        self.set_config_from_file("")
        self.create_default_config_file()

    def create_default_config_file(self):
        file_path = os.path.join(HOME, CONFIG_FILE)
        if os.path.exists(file_path):
            return
        with open(file_path, "w") as file:
            yaml.dump(DEFAULT_CONFIG, file)

    def set_config_from_file(self, directory: str):
        file_path = os.path.join(directory, CONFIG_FILE)
        config_data = {}
        try:
            with open(file_path, "r") as file:
                config_data = yaml.load(file, Loader=yaml.FullLoader)
        except OSError:
            pass
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
