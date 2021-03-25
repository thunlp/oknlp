from ..config import config


class BaseAlgorithm:
    """算法类的基类，派生类需要实现__call__(self, sents)函数

    有类属性self.device，表示模型需要放在哪个device上
    """
    def __init__(self, device=None):
        if device is None:
            device = config.default_device
        self.to(device)

    def to(self, device: str):
        """将模型转移到device上，返回self
        """
        self.device = device
        return self

    def __call__(self, sents: list) -> list:
        return sents
