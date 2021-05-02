from typing import List
from .BaseTyping import BaseTyping
from .bert_typing import BertTyping


def get_by_name(name: str = "") -> BaseTyping:
    """根据条件获取一个Typing类的实例，无法根据条件获取时返回BertTyping()

    :param str name: Typing类使用到的方法

        * "bert"->返回以Bert模型实现的算法

        * 默认返回以Bert模型实现的算法

    :returns: 一个Typing类的实例
    """
    name = name.lower()
    if name == "bert":
        return BertTyping()
    return BertTyping()


def get_all() -> List[BaseTyping]:
    """获取所有Typing类的实例
    """
    return [BertTyping()]
