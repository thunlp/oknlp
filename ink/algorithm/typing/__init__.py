from .bert_typing import BertTyping


def get_typing(name: str = ""):
    """根据条件获取一个Typing类的实例，无法根据条件获取时返回BertTyping()
    """
    name = name.lower()
    if name == "bert":
        return BertTyping()
    return BertTyping()


def get_all_typing() -> list:
    """获取所有Typing类的实例
    """
    return [BertTyping()]
