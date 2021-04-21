import torch.utils.data as Data
import torch
from transformers import BertTokenizer as tk


class Tokenizer:
    def __init__(self):
        self.tokenizer = tk.from_pretrained("bert-base-chinese")

    def __call__(self, text):
        return self.tokenizer.tokenize(text)

    def token2id(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class Dataset(Data.Dataset):
    tokenizer = Tokenizer()

    def __init__(self, sents):
        self.x = []
        self.y = []
        self.at = []
        max_length = 128
        for x in sents:
            sx = ['[CLS]']
            sy = [-1]
            sat = [1]
            for w in x:
                tokens = self.tokenizer(w)
                sx += tokens
                sy += [0] * len(tokens)
                sat += [1] * len(tokens)
            if len(sx) > max_length - 1:
                sx = sx[:max_length - 1]
                sy = sy[:max_length - 1]
                sat = sat[:max_length - 1]
            sx += ['[SEP]']
            sy += [-1]
            sat += [1]  # mask
            sx = self.tokenizer.token2id(sx)
            sx += [0] * (max_length - len(sx))
            sy += [-1] * (max_length - len(sy))
            sat += [0] * (max_length - len(sat))
            self.x.append(sx)
            self.y.append(sy)
            self.at.append(sat)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.LongTensor(self.x[i]), torch.LongTensor(self.y[i]), torch.LongTensor(self.at[i])
