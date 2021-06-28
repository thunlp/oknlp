from typing import List, Tuple
from ..BaseAlgorithm import BaseAlgorithm


class BasePosTagging(BaseAlgorithm):
    def __call__(self, sents: List[str]) -> List[List[Tuple[str, str]]]:
        """
        Args:
            sents: 输入的句子列表。
        Returns:
            返回一个和输入列表长度相同的列表，其中每一项对应输入的词性标注结果。
        
        更多信息请参考 :doc:`词性标注 - 示例</examples/postagging>`

        Examples:
            >>> import oknlp
            >>> postagging = oknlp.postagging.get_by_name()
            >>> postagging(['我爱北京天安门', '今天天气真好'])
            [
                [('我', 'PN'), ('爱', 'VV'), ('北京', 'NR'), ('天安门', 'NR')],
                [('今天', 'NT'), ('天气', 'NN'), ('真', 'AD'), ('好', 'VA')]
            ]

        """
        return super().__call__(sents)
