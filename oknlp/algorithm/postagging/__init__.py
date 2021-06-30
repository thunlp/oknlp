from typing import List
from .BasePosTagging import BasePosTagging
from .bert_postagging import BertPosTagging


def get_by_name(name: str = "", **kwargs) -> BasePosTagging:
    """
    目前支持的算法：

    * bert
    
    """
    name = name.lower()
    if name == "bert":
        return BertPosTagging(**kwargs)
    return BertPosTagging(**kwargs)


def get_all(**kwargs) -> List[BasePosTagging]:
    """获取所有PosTagging类的实例
    """
    return [BertPosTagging(**kwargs)]
