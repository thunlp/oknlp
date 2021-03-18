import torch
import os
from ..preProcessor import SeqDataset
from ...nn.models import BertLSTMCRF
from ...data import  load
from ...config import config

def get_word(path, tag_map):
    results = []
    record = {}
    for index, tag_id in enumerate(path):
        if tag_id == 0:
            continue
        if tag_id == 1:
            if ('begin' in record):
                record['end'] = record['begin']
                results.append(record)
                record = {}
                record['begin'] = index
            else:
                record['begin'] = index
        else:
            if ('begin' in record):
                record['end'] = index
                results.append(record)
                record = {}
            else:
                results.append({'begin': index, 'end': index})
                record = {}

    if (record.get('begin')):
        record['end'] = len(path) - 1
        results.append(record)
        record = {}
    return results

class ChineseWordSegmentation:
    def __init__(self, device=None):
        self.cws_path = load('cws')
        self.basic_path = load('basic')
        self.model = BertLSTMCRF(input_size=300, hidden_size=200, label_sizes=[3], toplayer='CRF')
        self.seq = SeqDataset( embedding_path=os.path.join(self.basic_path, 'sgns300'),
                         vocab_path=os.path.join(self.basic_path, 'vocab.pkl'),
                         tag_path=os.path.join(self.cws_path, 'tagset_cws.txt'),
                         bert_tokenizer='bert-base-chinese')

        self.id2tag = self.seq.tagging()
        self.checkpoint = torch.load(os.path.join(self.cws_path, "cws.pth"), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.checkpoint['net'], False)
        self.model.eval()

        if device is None:
            device = config.default_device
        self.to(device)
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def cws(self,sents):
        results = []
        for sent in sents:

            test_pkg = {'token': sent, 'tag': ' '.join(['0'] * len(sent))}
            tokens, tags, mask = self.seq.transformer.item2id(test_pkg)
            tokens = torch.LongTensor([tokens])
            result = ''

            with torch.no_grad():
                out = self.model.predict(3, tokens.to(self.device), mask.to(self.device))
                out = out.cpu().tolist()
                out_etts = [get_word(line, self.id2tag) for line in out]

                for seg in out_etts[0]:
                    result += sent[seg['begin']:seg['end'] + 1] + ' '
                reuslt = result[:-1]
            results.append(reuslt)
        return results

