from typing import List, Dict, Union
from ..BaseAlgorithm import BaseAlgorithm


class BaseNER(BaseAlgorithm):
    """命名实体分类(NamedEntityRecognition)算法的基类，所有的NER算法需要继承该类并实现preprocess(self, x)、infer(self, batch)、postprocess(self, x)方法

    该基类本身并不实现任何算法，你可以通过调用该模块下的get_函数获取有具体实现的算法类
    """
    def __call__(self, sents: List[str]) -> List[List[Dict[str, Union[str, int]]]]:
        """
        :param List[str] sents: 要进行命名实体分类的字符串列表
        :return: List[List[Dict[str, Union[str, int]]]] 每句话的命名实体分类结果
        :example:
            >>> import oknlp
            >>> ner = oknlp.ner.get_by_name()
            >>> sents = ['我爱北京天安门']
            >>> ner(sents)
            [[{'type': 'LOC', 'begin': 2, 'end': 3}, {'type': 'LOC', 'begin': 4, 'end': 6}]]
        """
        return super().__call__(sents)
