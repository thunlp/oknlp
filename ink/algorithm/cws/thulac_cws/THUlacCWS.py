from ...BaseAlgorithm import BaseAlgorithm
from ...._C import THUlac


class THUlacCWS(BaseAlgorithm):
    def __init__(self, device=None):
        self.model = THUlac()
        super().__init__(device)

    def __call__(self, sents):
        return [self.model.cws(sent) for sent in sents]
