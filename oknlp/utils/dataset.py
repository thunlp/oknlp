import torch.utils.data as Data
import torch


class Dataset(Data.Dataset):
    def __init__(self, sents, tokenizer, max_length=128):
        self.sents = sents
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        sent = self.sents[i]
        tokens = self.tokenizer.tokenize(sent)[:self.max_length-2]
        sx = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]']) + [0] * (self.max_length - 2 - len(tokens))
        sy = [-1] + [0] * len(tokens) + [-1] * (self.max_length - 1 - len(tokens))
        sat = [1] * (len(tokens) + 2) + [0] * (self.max_length - 2 - len(tokens))
        return torch.LongTensor(sx), torch.LongTensor(sy), torch.LongTensor(sat)
