from .bert_postagging import BertPosTagging


def get_pos_tagging(name: str = ""):
    """根据条件获取一个PosTagging类的实例，无法根据条件获取时返回BertPosTagging()
    """
    name = name.lower()
    if name == "bert":
        return BertPosTagging()
    return BertPosTagging()


def get_all_pos_tagging() -> list:
    """获取所有PosTagging类的实例
    """
    return [BertPosTagging()]
