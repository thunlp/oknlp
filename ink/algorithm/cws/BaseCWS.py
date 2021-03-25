from ..BaseAlgorithm import BaseAlgorithm


class BaseCWS(BaseAlgorithm):
    """中文分词(ChineseWordSegmentation)算法的基类，所有的CWS算法需要继承该类并实现__call__(self, sents)函数

    该基类本身并不实现任何算法，你可以通过调用该模块下的get_函数获取有具体实现的算法类
    """
    def __init__(self, device):
        super().__init__(device)

    def to(self, device: str):
        return super().to(device)

    def __call__(self, sents: list[str]) -> list[str]:
        """
        Args:
            sents: list[str]
                表示需要进行分词的字符串列表，例如['今天天气真好', '我爱北京天安门']

        Returns:
            list[str]
                表示每句话分词后的结果（词语间用空格隔开），例如['今天 天气 真 好', '我 爱 北京 天安门']
        """
        return super().__call__(sents)
