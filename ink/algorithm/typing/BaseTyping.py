from typing import List, Tuple
from ..BaseAlgorithm import BaseAlgorithm


class BaseTyping(BaseAlgorithm):
    """细粒度实体分类(Typing)算法的基类，所有的Typing算法需要继承该类并实现__call__(self, sents)函数

    该基类本身并不实现任何算法，你可以通过调用该模块下的get_函数获取有具体实现的算法类
    """
    def __init__(self, device=None):
        super().__init__(device)

    def to(self, device: str):
        return super().to(device)

    def __call__(self, sents: List[Tuple[str, Tuple[int, int]]]) -> List[List[Tuple[str, float]]]:
        """
        Args:
            sents: List[Tuple[str, list]]
                每一个输入都是一个[str, Tuple[int, int]]元组，第一个参数是上下文字符串，第二个参数是实体的提及位置[begin, end)，例如，
                [("3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", (30, 33))]

        Returns:
            List[List[Tuple[str, float]]]
                对每一个输入，输出所有可能的类型及对应的分数，例如，
                [[('object', 0.35983458161354065), ('event', 0.8602959513664246), ('attack', 0.12778696417808533), ('disease', 0.2171688675880432)]]
        """
        return super().__call__(sents)
