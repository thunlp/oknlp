from typing import List
from .BaseCWS import BaseCWS
from .thulac_cws import THUlacCWS
from .bert_cws import BertCWS

def get_by_name(name: str = "", **kwargs) -> BaseCWS:
    """根据条件获取一个CWS类的实例，无法根据条件获取时返回BertCWS()

    :param str name: CWS类使用到的方法

        * "bert"->返回以Bert模型实现的算法

        * "thulac"->返回以THUlac实现的算法

        * 默认返回以Bert模型实现的算法

    :returns: 一个CWS类的实例
    """
    name = name.lower()
    if name == "bert":
        return BertCWS(**kwargs)
    if name == "thulac":
        return THUlacCWS(**kwargs)
    return BertCWS(**kwargs)


def get_all(**kwargs) -> List[BaseCWS]:
    """获取所有CWS类的实例
    """
    return [BertCWS(**kwargs), THUlacCWS(**kwargs)]
