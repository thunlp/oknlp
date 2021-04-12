from typing import List
from .BaseTyping import BaseTyping
from .bert_typing import BertTyping


def get_typing(name: str = "") -> BaseTyping:
    """根据条件获取一个Typing类的实例，无法根据条件获取时返回BertTyping()

    Args:
        name: str，表示Typing类使用到的方法
            "bert"->返回以Bert模型实现的算法
    """
    name = name.lower()
    if name == "bert":
        return BertTyping()
    return BertTyping()


def get_all_typing() -> List[BaseTyping]:
    """获取所有Typing类的实例
    """
    return [BertTyping()]
