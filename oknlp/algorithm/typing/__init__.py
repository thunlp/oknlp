from typing import List
from .BaseTyping import BaseTyping
from .bert_typing import BertTyping


def get_by_name(name: str = "", **kwargs) -> BaseTyping:
    """
    目前支持的算法：

    * bert
    
    """
    name = name.lower()
    if name == "bert":
        return BertTyping(**kwargs)
    return BertTyping(**kwargs)


def get_all(**kwargs) -> List[BaseTyping]:
    """获取所有Typing类的实例
    """
    return [BertTyping(**kwargs)]
