import os
import torch
import json
from transformers import BertTokenizer
from ink.nn.models.bertlinearsigmoid import BertLinearSigmoid
from ink.data import load


class Typing:
    def __init__(self):
        typ_path = load('typ')
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(typ_path, 'vocab.txt'), additional_special_tokens=['[ENL]', '[ENR]'])
        self.types = json.loads(open(os.path.join(typ_path, 'types.json'), 'r').read())
        self.model = BertLinearSigmoid()
        self.model.load_state_dict(torch.load(os.path.join(typ_path, 'typing.pth'), map_location=lambda storage, loc: storage))

    def get_input(self, text: str):
        text = self.tokenizer.tokenize(text)
        pos = text.index('[ENL]')
        text = self.tokenizer.convert_tokens_to_ids(text)
        text = torch.tensor([text])
        return text, pos

    def __call__(self, sents: list) -> list:
        """
        Args:
            sents: list[[str, list]], which means [[string, span]], span is [begin, end). For example,
                [["3月15日,北方多地正遭遇近10年来强度最大、影响范围最广的沙尘暴。", [30, 33]]]

        Return:
            list[(str, float)], which means [(type, score)]. For example,
                [[('object', 0.35983458161354065), ('event', 0.8602959513664246), ('attack', 0.12778696417808533), ('disease', 0.2171688675880432)]]
        """
        results = []
        for (text, span) in sents:
            text = text[:span[0]] + '[ENL]' + text[span[0]: span[1]] + '[ENR]' + text[span[1]:]
            text, pos = self.get_input(text)
            out = self.model(text, pos)
            out = out.tolist()
            result = []
            for i, score in enumerate(out):
                if score > 0.1:
                    result.append((self.types[i], score))
            results.append(result)
        return results
