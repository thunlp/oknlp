from typing import List
from .BaseCWS import BaseCWS
from .thulac_cws import THUlacCWS
from .bert_cws import BertCWS 

def get_by_name(name: str = "", **kwargs) -> BaseCWS:
    """
    目前支持的算法：

    * bert
    * thulac
    
    """
    name = name.lower()
    if name == "bert":
        return BertCWS(**kwargs)
    if name == "thulac":
        return THUlacCWS(**kwargs)
    return BertCWS(**kwargs)


def get_all(**kwargs) -> List[BaseCWS]:
    return [BertCWS(**kwargs), THUlacCWS(**kwargs)]
