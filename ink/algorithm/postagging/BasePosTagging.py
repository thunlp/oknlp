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
        Args:
            sents: List[str]
                表示需要进行词性标注的字符串列表，例如['清华大学自然语言处理与社会人文计算实验室', '今天天气真好']

        Returns:
            List[List[Tuple[str, str]]]
                对于每一个输入的字符串，返回一个List[Tuple[str, str]]，Tuple[0]表示一个词，Tuple[1]表示这个词对应的词性，例如
                [
                    [('清华', 'NR'), ('大学', 'NN'), ('自然', 'NN'), ('语言', 'NN'), ('处理', 'NN'), ('与', 'CC'), ('社会', 'NN'), ('人文', 'NN'), ('计算', 'NN'), ('实验室', 'NN')],

                    [('今天', 'NT'), ('天气', 'NN'), ('真', 'AD'), ('好', 'VA')]
                ]
        """
        return super().__call__(sents)
