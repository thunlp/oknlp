from .bert_ner import BertNER


def get_ner(name: str = ""):
    """根据条件获取一个NER类的实例，无法根据条件获取时返回BertNER()
    """
    name = name.lower()
    if name == "bert":
        return BertNER()
    return BertNER()


def get_all_ner() -> list:
    """获取所有NER类的实例
    """
    return [BertNER()]
