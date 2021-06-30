from typing import List
from .BaseNER import BaseNER
from .bert_ner import BertNER


def get_by_name(name: str = "", **kwargs) -> BaseNER:
    """
    目前支持的算法：

    * bert
    
    """
    name = name.lower()
    if name == "bert":
        return BertNER(**kwargs)
    return BertNER(**kwargs)


def get_all(**kwargs) -> List[BaseNER]:
    """获取所有NER类的实例
    """
    return [BertNER(**kwargs)]
