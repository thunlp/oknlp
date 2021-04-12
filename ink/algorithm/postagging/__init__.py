from typing import List
from .BasePosTagging import BasePosTagging
from .bert_postagging import BertPosTagging


def get_pos_tagging(name: str = "") -> BasePosTagging:
    """根据条件获取一个PosTagging类的实例，无法根据条件获取时返回BertPosTagging()

    Args:
        name: str，表示PosTagging类使用到的方法
            "bert"->返回以Bert模型实现的算法
    """
    name = name.lower()
    if name == "bert":
        return BertPosTagging()
    return BertPosTagging()


def get_all_pos_tagging() -> List[BasePosTagging]:
    """获取所有PosTagging类的实例
    """
    return [BertPosTagging()]
