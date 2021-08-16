from typing import List
from ...._C import THUlac
from ....data import load
from ..BaseCWS import BaseCWS
from ....utils import DictExtraction
from ....utils.format_output import format_output, dict_format
from functools import reduce

labels = reduce(lambda x, y: x+y, [[f"{kd}-{l}" for kd in ('B','I','O')] for l in ('SEG',)])
class THUlacCWS(BaseCWS):
    """基于THULAC的分词算法

    :Name: thulac

    更多信息请参考 thulac 文档： `http://thulac.thunlp.org/ <http://thulac.thunlp.org/>`_ 。

    **示例**

    .. code-block:: python

        oknlp.cws.get_by_name("thulac")
    """

    def __init__(self, dictionary = []):
        model_path = load("cws.lac", 'fp32')
        self.model = THUlac(model_path)
        self.__closed = False
        self.keyword_processor=DictExtraction(case_sensitive = False)
        self.keyword_processor.add_keywords_from_list(dictionary)            

    def __call__(self, sents: List[str]) -> List[List[str]]:
        result = [self.model.cut(sent) for sent in sents]
        results = []
        for sep in result:
            if sep[-1] == '\n':
                sep = sep[:-1]
            results.append(sep)
        results = [dict_format(result, self.keyword_processor.extract_dictwords(''.join(result))) for result in results]
         
        return results
    def close(self):
        if self.__closed:
            return
        self.__closed = True
        del self.model