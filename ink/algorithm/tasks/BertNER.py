import torch
import os
from ..preProcessor import SeqDataset
from ink.nn.models.bertlstmcrf import BERT_LSTM
from ink.utils.format import get_entity
from ink.data import  load

torch.manual_seed(2018)

class NamedEntityRecognition:
    def __init__(self):
        self.ner_path = load('ner')
        self.basic_path = load('basic')
        self.model = BERT_LSTM(input_size=300, hidden_size=200, label_sizes=[8], toplayer='CRF')
        self.seq = SeqDataset( embedding_path=os.path.join(self.basic_path, 'sgns300'),
                         vocab_path=os.path.join(self.basic_path, 'vocab.pkl'),
                         tag_path=os.path.join(self.ner_path, 'tagset.txt'),
                         bert_tokenizer='bert-base-chinese')
        self.id2tag = self.seq.tagging()
        self.checkpoint = torch.load(os.path.join(self.ner_path, "bert2.pth"), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.checkpoint['net'], False)
    def ner(self,sents):
        result = []
        for sent in sents:
            result = []
            test_pkg = {'token': sent, 'tag': ' '.join(['O'] * len(sent))}
            tokens, tags, masks = self.seq.transformer.item2id(test_pkg)
            tokens = torch.LongTensor([tokens])
            with torch.no_grad():
                out = self.model.predict(8, tokens, masks)
                out = out.numpy().tolist()
                out_etts = [get_entity(line, self.id2tag) for line in out]
                result.append(out_etts)
        return result

