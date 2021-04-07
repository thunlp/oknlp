import re
import torch
from transformers import BertTokenizer


def sequence_mask(sequence_length, max_len=80, device=None, padding=True):  # sequence_length :(batch_size, )
    if (not padding):
        max_len = torch.max(sequence_length)
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


class SeqDataset:
    def __init__(self, tag_path, bert_tokenizer=None, max_sentence_length=80):
        self.bert_tokenizer = bert_tokenizer
        self.transformer = SeqTransformer(tag_path, bert_tokenizer, max_sentence_length)
        with open(tag_path, 'r', encoding='utf-8') as f_in:
            tagset = re.split(r'\s+', f_in.read().strip())
            self.tag2id = dict((tag, idx + 1) for idx, tag in enumerate(tagset))  # 0 is padding
            self.id2tag = dict((idx + 1, tag) for idx, tag in enumerate(tagset))

    def tagging(self):
        return self.id2tag

    def __getitem__(self, index):
        return (self.tokens[index], self.tags[index], self.lengths[index])


class SeqTransformer:
    def __init__(self, tag_path, bert_tokenizer=None, max_sentence_length=80):
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        with open(tag_path, 'r', encoding='utf-8') as f_in:
            tagset = re.split(r'\s+', f_in.read().strip())
            self.tag2id = dict((tag, idx + 1) for idx, tag in enumerate(tagset))  # 0 is padding
            self.id2tag = dict((idx + 1, tag) for idx, tag in enumerate(tagset))
        self.padding = lambda x: [(0 if idx >= len(x) else x[idx]) for idx in range(max_sentence_length)]

    def tagging(self):
        return self.id2tag

    def item2id(self, item):
        l = len(item['token'])
        tokens = self.padding(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(item['token'])))
        tags = self.padding([self.tag2id[t] for t in (item['tag'].split(' '))])
        lens = torch.LongTensor([l])
        masks = sequence_mask(lens, device=None)
        return tokens, tags, masks
