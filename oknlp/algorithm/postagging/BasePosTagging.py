from typing import List, Tuple
from ..BaseAlgorithm import BaseAlgorithm


class BasePosTagging(BaseAlgorithm):
    """词性标注(PosTagging)算法的基类，所有的PosTagging算法需要继承该类并实现__call__(self, sents)函数

    该基类本身并不实现任何算法，你可以通过调用该模块下的get_函数获取有具体实现的算法类
    """
    def __init__(self, device=None):
        super().__init__(device)

    def to(self, device: str):
        return super().to(device)

    def __call__(self, sents: List[str]) -> List[List[Tuple[str, str]]]:
        """
        :param List[str] sents: 需要进行词性标注的字符串列表
        :return: List[List[Tuple[str, str]]] 每句话中每个词及其对应的词性
        :example:
            >>> import oknlp
            >>> postagging = oknlp.postagging.get_by_name()
            >>> sents = ['我爱北京天安门', '今天天气真好']
            >>> postagging(sents)
            [
                [('我', 'PN'), ('爱', 'VV'), ('北京', 'NR'), ('天安门', 'NR')],
                [('今天', 'NT'), ('天气', 'NN'), ('真', 'AD'), ('好', 'VA')]
            ]
        """
        return super().__call__(sents)
