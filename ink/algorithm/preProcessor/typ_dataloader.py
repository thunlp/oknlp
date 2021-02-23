import os
import re
import numpy as np
from .pre_processor import GeneralTransformer


def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask


def multilb(shape, input):
    label = [0] * shape
    for idx in input:
        label[idx] = 1
    return label


class TypDataset:
    def __init__(self, file_path, vocab_path, embedding_path, type_path, bert_tokenizer=None, max_sentence_length=80,
                 dir_name_given=None):
        self.tokens, self.types, self.lengths, self.masks = [], [], [], []
        # translation
        self.dir_name_given = dir_name_given
        self.bert_tokenizer = bert_tokenizer
        self.transformer = TypTransformer(vocab_path, embedding_path, type_path, bert_tokenizer=bert_tokenizer,
                                              max_sentence_length=80)

    def save(self):

        if (self.bert_tokenizer is not None):
            dir_name = "TypCached(BERT)"
        else:
            dir_name = "TypCached"
        if (self.dir_name_given is not None):
            dir_name = self.dir_name_given
        try:
            os.mkdir(dir_name)
        except:
            pass
        np.save(os.path.join(dir_name, "tokens.npy"), self.tokens)
        np.save(os.path.join(dir_name, "types.npy"), self.types)
        np.save(os.path.join(dir_name, "masks.npy"), self.masks)

    def load(self):

        if (self.bert_tokenizer is not None):
            dir_name = "TypCached(BERT)"
        else:
            dir_name = "TypCached"
        if (self.dir_name_given is not None):
            dir_name = self.dir_name_given
        self.tokens = np.load(os.path.join(dir_name, "tokens.npy"))
        self.types = np.load(os.path.join(dir_name, "types.npy"))
        self.masks = np.load(os.path.join(dir_name, "masks.npy"))

    def __getitem__(self, index):
        return (self.tokens[index], self.types[index], self.masks[index])

    def __len__(self):
        return len(self.tokens)


class TypTransformer(GeneralTransformer):
    def __init__(self, vocab_path, embedding_path, type_path, bert_tokenizer=None, max_sentence_length=80):
        GeneralTransformer.__init__(self, vocab_path, embedding_path, bert_tokenizer)
        self.bert_tokenizer = bert_tokenizer
        self.max_sentence_length = max_sentence_length
        with open(type_path, 'r', encoding='utf-8') as f_in:
            tagset = re.split(r'\s+', f_in.read().strip())
            self.types = dict((tag, idx) for idx, tag in enumerate(tagset))

        self.padding = lambda x: [(0 if idx >= len(x) else x[idx]) for idx in range(max_sentence_length)]
        self.emb_padding = lambda x: [(self.embedding_matrix[self.vocab['<unk>']] if idx >= len(x) else x[idx]) for idx
                                      in range(max_sentence_length)]

    def item2id(self, item):
        try:
            type_num = [self.types[name] for name in item['types']]
        except:
            # already nums
            type_num = item['types']
        type_num = multilb(len(self.types.keys()), type_num)

        sentence = item['text']
        pos = item['span']

        if self.bert_tokenizer:
            tokens = self.tokenizer.tokenize(sentence)
        else:
            tokens = list(sentence)

        if self.bert_tokenizer:
            tokens = self.padding(self.tokenizer.convert_tokens_to_ids(tokens))
        else:
            tokens = [self._word_to_emb(word=word) for word in item['text']]
            tokens = self.emb_padding(tokens)
        context = [0, len(sentence)]
        masks = [convert_pos_to_mask(pos, self.max_sentence_length),
                 convert_pos_to_mask(context, self.max_sentence_length)]
        return tokens, type_num, masks

    def _word_to_emb(self, word):
        return self.embedding_matrix[self.vocab[word]] if word in self.vocab else self.embedding_matrix[
            self.vocab['<unk>']]


