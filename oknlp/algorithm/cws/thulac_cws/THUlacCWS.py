from ...._C import THUlac
from ....data import load


class THUlacCWS:
    """基于THULAC的分词算法

    :Name: thulac

    更多信息请参考 thulac 文档： `http://thulac.thunlp.org/ <http://thulac.thunlp.org/>`_ 。

    **示例**

    .. code-block:: python

        oknlp.cws.get_by_name("thulac")
    """

    def __init__(self):
        model_path = load("cws.lac", 'fp32')
        self.model = THUlac(model_path)

    def __call__(self, sents):
        result = [self.model.cut(sent) for sent in sents]
        results = []
        for sep in result:
            if sep[-1] == '\n':
                sep = sep[:-1]
            results.append(sep)
        return results
