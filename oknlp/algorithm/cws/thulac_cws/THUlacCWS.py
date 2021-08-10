from typing import List
from ...._C import THUlac
from ....data import load
from ..BaseCWS import BaseCWS
from ....utils import DictExtraction
from ....utils.format_output import format_output
from functools import reduce

labels = reduce(lambda x, y: x+y, [[f"{kd}-{l}" for kd in ('B','I','O')] for l in ('SEG',)])
class THUlacCWS(BaseCWS):
    """基于THULAC的分词算法

    :Name: thulac

    更多信息请参考 thulafc 文档： `http://thulac.thunlp.org/ <http://thulac.thunlp.org/>`_ 。

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

    def _seg_new_word(self, seg_list):
        sent = ''.join(seg_list)
        words = self.keyword_processor.extract_dictwords(sent)
        seps = []
        tag = 0
        count = 0
        tag_end = 0
        for seg in seg_list:
            if tag_end != 0:
                if tag_end < len(seg):
                    seps.append(seps.pop() + seg[ : tag_end])
                    seg = seg[tag_end : ]
                    count += tag_end
                    tag_end = 0
                else:
                    seps.append(seps.pop() + seg[ : tag_end])
                    tag_end -= len(seg)
                    count += len(seg)
                    continue
            for idx in range(count,count+len(seg)):
                if idx in words:
                    tag = 1
                    seps.append(seg[ : idx-count-1])
                    if count+len(seg) > words[idx]:
                        seps.append(seg[idx-count: words[idx]-count])
                    else:
                        seps.append(seg[idx-count: ])
                        tag_end = words[idx] - (count + len(seg)-1)
            if tag ==0:
                seps.append(seg)
            count += len(seg)
            tag =0
        return seps
            

    def __call__(self, sents: List[str]) -> List[List[str]]:
        result = [self.model.cut(sent) for sent in sents]
        results = []
        for sep in result:
            if sep[-1] == '\n':
                sep = sep[:-1]
            results.append(sep)
        results = [self._seg_new_word(result) for result in results]
         
        return results
    def close(self):
        if self.__closed:
            return
        self.__closed = True
        del self.model