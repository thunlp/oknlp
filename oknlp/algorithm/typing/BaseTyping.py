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
        :param List[Tuple[str,Tuple[int,int]]] sents: 上下文，实体提及位置[begin, end)的列表
        :return: List[List[Tuple[str, float]]] 每一个实体所有可能的类型及对应的分数
        :example:
            >>> import oknlp
            >>> typing = oknlp.typing.get_by_name()
            >>> sents = [("3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", (30, 33))]
            >>> typing(sents)
            [[('object', 0.26066625118255615), ('event', 0.9411928653717041)]]
        """
        return super().__call__(sents)
