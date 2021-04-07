from typing import List
from .BaseCWS import BaseCWS
from .bert_cws import BertCWS
from .thulac_cws import THUlacCWS


def get_cws(name: str = "") -> BaseCWS:
    """根据条件获取一个CWS类的实例，无法根据条件获取时返回BertCWS()

    Args:
        name: str，表示CWS类使用到的方法，目前支持: "bert", "thulac"
    """
    name = name.lower()
    if name == "bert":
        return BertCWS()
    if name == "thulac":
        return THUlacCWS()
    return BertCWS()


def get_all_cws() -> List[BaseCWS]:
    """获取所有CWS类的实例
    """
    return [BertCWS(), THUlacCWS()]
