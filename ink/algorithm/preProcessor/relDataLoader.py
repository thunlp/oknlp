import os
import re
import torch
import numpy as np
from .preProcessor import GeneralDataset, GeneralTransformer


def convert_pos_to_mask(e_pos, max_len=128):
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask

class RelDataset(GeneralDataset):
    def __init__(self, file_path, vocab_path, embedding_path, rel_path, bert_tokenizer=None, max_sentence_length=80,
                 dir_name_given=None):
        GeneralDataset.__init__(self, file_path)
        self.tokens, self.rels, self.lengths, self.masks = [], [], [], []
        # translation
        self.dir_name_given = dir_name_given
        self.bert_tokenizer = bert_tokenizer
        try:
            self.load()
        except:
            self.transformer = RelTransformer(vocab_path, embedding_path, rel_path, bert_tokenizer=bert_tokenizer,
                                              max_sentence_length=80)
            for item in self.input:
                #   hrts = []
                try:
                    # one line one pair
                    if (len(item['text']) > 80):
                        continue
                    to, re, mask = self.transformer.item2id(item)

                    self.tokens.append(to)
                    self.rels.append(re)
                    self.lengths.append(len(to))
                    self.masks.append(mask)
                except:
                    pass
                # transform
            self.tokens = np.array(self.tokens)
            self.tags = np.array(self.tags)
            self.lengths = np.array(self.lengths)
            self.masks = np.array(self.masks)
            self.tokens = self.tokens.astype(np.int64)
            self.save()

    def save(self):

        if (self.bert_tokenizer is not None):
            dir_name = "RelCached(BERT)"
        else:
            dir_name = "RelCached"
        if (self.dir_name_given is not None):
            dir_name = self.dir_name_given
        try:
            os.mkdir(dir_name)
        except:
            pass
        np.save(dir_name + "/tokens.npy", self.tokens)
        np.save(dir_name + "/relations.npy", self.rels)
        np.save(dir_name + "/masks.npy", self.masks)

    def load(self):

        if (self.bert_tokenizer is not None):
            dir_name = "RelCached(BERT)"
        else:
            dir_name = "RelCached"
        if (self.dir_name_given is not None):
            dir_name = self.dir_name_given
        self.tokens = np.load(dir_name + "/tokens.npy")
        self.rels = np.load(dir_name + "/relations.npy")
        self.masks = np.load(dir_name + "/masks.npy")

    def __getitem__(self, index):
        return (self.tokens[index], self.rels[index], self.masks[index])

    def __len__(self):
        return len(self.tokens)


class RelTransformer(GeneralTransformer):
    def __init__(self, vocab_path, embedding_path, rel_path, bert_tokenizer=None, max_sentence_length=80):
        GeneralTransformer.__init__(self, vocab_path, embedding_path, bert_tokenizer)
        self.bert_tokenizer = bert_tokenizer
        self.max_sentence_length = max_sentence_length
        with open(rel_path, 'r', encoding='utf-8') as f_in:
            tagset = re.split(r'\s+', f_in.read().strip())
            self.relations = dict((tag, idx) for idx, tag in enumerate(tagset))

        self.padding = lambda x: [(0 if idx >= len(x) else x[idx]) for idx in range(max_sentence_length)]
        self.emb_padding = lambda x: [(self.embedding_matrix[self.vocab['<unk>']] if idx >= len(x) else x[idx]) for idx
                                      in range(max_sentence_length)]

    def item2id(self, item):
        rel_num = self.relations[item['relation']]
        sentence = item['text']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            pos_min = pos_head
            pos_max = pos_tail
            rev = False
        if self.bert_tokenizer:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = list(sentence[:pos_min[0]])
            ent0 = list(sentence[pos_min[0]:pos_min[1]])
            sent1 = list(sentence[pos_min[1]:pos_max[0]])
            ent1 = list(sentence[pos_max[0]:pos_max[1]])
            sent2 = list(sentence[pos_max[1]:])
        self.mask_entity = False
        if rev:
            if self.mask_entity:
                ent0 = ['[unused6]']
                ent1 = ['[unused5]']
            pos_tail = [len(sent0), len(sent0) + len(ent0)]
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        else:
            if self.mask_entity:
                ent0 = ['[unused5]']
                ent1 = ['[unused6]']
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        tokens = sent0 + ent0 + sent1 + ent1 + sent2

        if self.bert_tokenizer:
            tokens = self.padding(self.tokenizer.convert_tokens_to_ids(tokens))
        else:
            tokens = [self._word_to_emb(word=word) for word in item['text']]
            tokens = self.emb_padding(tokens)
        context = [0, len(sentence)]
        masks = [convert_pos_to_mask(pos_head, self.max_sentence_length),
                 convert_pos_to_mask(pos_tail, self.max_sentence_length),
                 convert_pos_to_mask(context, self.max_sentence_length)]

        return tokens, rel_num, masks

    def _word_to_emb(self, word):
        return self.embedding_matrix[self.vocab[word]] if word in self.vocab else self.embedding_matrix[
            self.vocab['<unk>']]