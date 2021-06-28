from typing import List, Tuple
from ..BaseAlgorithm import BaseAlgorithm


class BaseTyping(BaseAlgorithm):
    def __call__(self, sents: List[Tuple[str, Tuple[int, int]]]) -> List[List[Tuple[str, float]]]:
        """
        Args:
            sents: 输入的列表，其中每一项由一个二元组组成，分别表示 “输入的句子”和“实体位置”。实体位置由一个左闭右开区间的二元组表示。
        Returns:
            返回一个和输入列表长度相同的列表，其中每一项表示对应输入的细粒度实体分类结果。
        
        更多信息请参考 :doc:`细粒度实体分类 - 示例</examples/typing>`

        Examples:
            >>> import oknlp
            >>> typing = oknlp.typing.get_by_name()
            >>> typing([("3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", (30, 33))])
            [[('object', 0.26066625118255615), ('event', 0.9411928653717041)]]
        """
        return super().__call__(sents)
