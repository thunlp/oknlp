import torch.utils.data as Data
from ....nn.layers.bert import BertTokenizer
import torch


class Dataset(Data.Dataset):
    def __init__(self, max_length=128, examples=None):
        self.examples = examples

        self.sent = []
        self.attn = []
        self.label = []
        for ex in self.examples:
            sent = ['[CLS]'] + ex[0]
            label = [-1] + ex[1]
            attn = [1] + [1] * len(ex[0])
            if len(sent) > max_length - 1:
                sent = sent[:max_length - 1]
                label = label[:max_length - 1]
                attn = attn[:max_length - 1]
            sent += ['[SEP]']
            label += [-1]
            attn += [1]
            sent += ['[PAD]'] * (max_length - len(sent))
            label += [-1] * (max_length - len(label))
            attn += [0] * (max_length - len(attn))
            self.sent.append(sent)
            self.attn.append(attn)
            self.label.append(label)

        self.tokenizer = BertTokenizer()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return \
            torch.LongTensor([self.tokenizer.token2id(x) for x in self.sent[i]]), \
            torch.LongTensor(self.attn[i]), \
            torch.LongTensor(self.label[i])
