from typing import List
from .BasePosTagging import BasePosTagging
from .bert_postagging import BertPosTagging


def get_by_name(name: str = "") -> BasePosTagging:
    """根据条件获取一个PosTagging类的实例，无法根据条件获取时返回BertPosTagging()

    :param str name: PosTagging类使用到的方法

        * "bert"->返回以Bert模型实现的算法

        * 默认返回以Bert模型实现的算法

    :returns: 一个PosTagging类的实例
    """
    name = name.lower()
    if name == "bert":
        return BertPosTagging()
    return BertPosTagging()


def get_all() -> List[BasePosTagging]:
    """获取所有PosTagging类的实例
    """
    return [BertPosTagging()]
