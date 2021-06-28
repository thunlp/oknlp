from typing import List, Dict, Union
from ..BaseAlgorithm import BaseAlgorithm


class BaseNER(BaseAlgorithm):
    def __call__(self, sents: List[str]) -> List[List[Dict[str, Union[str, int]]]]:
        """
        Args:
            sents: 输入的句子列表。
        Returns:
            返回一个和输入长度相同的列表，其中每一项对应其命名实体结果。

            每个命名实体识别结果会包含多个实体，每个实体由一个包含三个字段的字典表示：

            * type
            * begin
            * end

        更多信息请参考 :doc:`命名实体识别 - 示例</examples/ner>`
    
        Examples:
            >>> import oknlp
            >>> ner = oknlp.ner.get_by_name()
            >>> ner(['我爱北京天安门'])
            [[{'type': 'LOC', 'begin': 2, 'end': 4}, {'type': 'LOC', 'begin': 4, 'end': 7}]]

        """
        return super().__call__(sents)
