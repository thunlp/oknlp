from ...BasicAlgorithm import BasicAlgorithm
from ...._C import THUlac


class THUlacCWS(BasicAlgorithm):
    def __init__(self, device=None):
        self.model = THUlac()
        super().__init__(device)

    def __call__(self, sents: list) -> list:
        """
        Args:
            sents: list[str]
                表示需要进行分词的字符串列表，例如，['今天天气真好']

        Returns:
            ??
        """
        return [self.model.cws(sent) for sent in sents]
