import os
import torch
import json
import torch.utils.data as Data
from transformers import BertTokenizer
from ..BaseTyping import BaseTyping
from ....nn.models import BertLinearSigmoid
from ....data import load


class Dataset(Data.Dataset):
    def __init__(self, sents, tokenizer, max_length=128):
        self.sents = sents
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        [text, span] = self.sents[i]
        if span[0] > self.max_length:
            text = text[span[0]:]
            span = (0, span[1] - span[0])
        text = text[:span[0]] + '<ent>' + text[span[0]: span[1]] + '</ents>' + text[span[1]:]
        pos = [text.index('<ent>')]
        text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))[:self.max_length]
        text += [0] * (self.max_length - len(text))
        return torch.LongTensor(text), torch.LongTensor(pos)


class BertTyping(BaseTyping):
    """使用Bert模型实现的Typing算法
    """
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

    def __call__(self, sents):
        results = []
        dataset = Dataset(sents, self.tokenizer)
        dataloader = Data.DataLoader(dataset, batch_size=8, num_workers=0)
        for text, pos in dataloader:
            with torch.no_grad():
                outs = self.model(text.to(self.device), pos)
                outs = outs.cpu().tolist()
            for out in outs:
                result = []
                for i, score in enumerate(out):
                    if score > 0.1:
                        result.append((self.types[i], score))
                results.append(result)
        return results
