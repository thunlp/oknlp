from ..config import config


class BasicAlgorithm:
    def __init__(self, device=None):
        if device is None:
            device = config.default_device
        self.to(device)

    def to(self, device: str):
        self.device = device
        return self

    def __call__(self, sents: list) -> list:
        return []
