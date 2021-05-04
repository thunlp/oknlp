from typing import List
from .BaseNER import BaseNER
from .bert_ner import BertNER


def get_by_name(name: str = "") -> BaseNER:
    """根据条件获取一个NER类的实例，无法根据条件获取时返回BertNER()

    :param str name: NER类使用到的方法

        * "bert"->返回以Bert模型实现的算法

        * 默认返回以Bert模型实现的算法

    :returns: 一个NER类的实例
    """
    name = name.lower()
    if name == "bert":
        return BertNER()
    return BertNER()


def get_all() -> List[BaseNER]:
    """获取所有NER类的实例
    """
    return [BertNER()]
