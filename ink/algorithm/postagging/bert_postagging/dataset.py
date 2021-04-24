import torch.utils.data as Data
import torch


class Dataset(Data.Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        sent = ['[CLS]'] + ex[0][:self.max_length-2] + ['[SEP]'] + ['[PAD]'] * (self.max_length - 2 - len(ex[0]))
        label = [1] * min(len(ex[0]) + 2, self.max_length) + [0] * (self.max_length - 2 - len(ex[0]))
        attn = [-1] + ex[1][:self.max_length-2] + [-1] * (self.max_length - 1 - len(ex[1]))
        return torch.LongTensor([self.tokenizer.convert_tokens_to_ids(x) for x in sent]),\
            torch.LongTensor(label),\
            torch.LongTensor(attn)
