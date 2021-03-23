import os
import torch
import json
from transformers import BertTokenizer
from ....nn.models import BertLinearSigmoid
from ....data import load
from ...BasicAlgorithm import BasicAlgorithm


class BertTyping(BasicAlgorithm):
    def __init__(self, device=None):
        typ_path = load('typ')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<ent>", "</ent>"]})
        self.types = json.loads(open(os.path.join(typ_path, 'types.json'), 'r').read())
        self.model = BertLinearSigmoid(len(self.tokenizer))
        self.model.load_state_dict(torch.load(os.path.join(typ_path, 'typing.pth'), map_location=lambda storage, loc: storage))
        self.model.eval()

        super().__init__(device)

    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)

    def __call__(self, sents: list) -> list:
        """
        Args:
            sents: list[[str, list]]
                对于每一个输入，第一个参数是上下文字符串，第二个参数是实体的提及位置[begin, end)，例如，
                [["3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", [30, 33]]]

        Returns:
            list[(str, float)]
                对每一个输入，输出所有可能的类型及对应的分数，例如，
                [[('object', 0.35983458161354065), ('event', 0.8602959513664246), ('attack', 0.12778696417808533), ('disease', 0.2171688675880432)]]
        """
        results = []
        for [text, span] in sents:
            text = text[:span[0]] + '<ent>' + text[span[0]: span[1]] + '</ents>' + text[span[1]:]
            text, pos = self.get_input(text)
            with torch.no_grad():
                out = self.model(text.to(self.device), pos)
                out = out.view(-1).tolist()
            result = []
            for i, score in enumerate(out):
                if score > 0.1:
                    result.append((self.types[i], score))
            results.append(result)
        return results

    def get_input(self, text: str):
        text = self.tokenizer.tokenize(text)
        pos = [text.index('<ent>')]
        text = self.tokenizer.convert_tokens_to_ids(text)
        text = torch.LongTensor([text])
        return text, pos
