import torch
import os
from ..preProcessor import SeqDataset
from ink.nn.models.bertlstmcrf import BERT_LSTM
from ink.data import  load

torch.manual_seed(2018)

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
    def __init__(self):
        self.cws_path = load('cws')
        self.basic_path = load('basic')
        self.model = BERT_LSTM(input_size=300, hidden_size=200, label_sizes=[3], toplayer='CRF')
        self.seq = SeqDataset(file_path=os.path.join(self.cws_path, "cws.txt"),
                         embedding_path=os.path.join(self.basic_path, 'sgns300'),
                         vocab_path=os.path.join(self.basic_path, 'vocab.pkl'),
                         tag_path=os.path.join(self.cws_path, 'tagset_cws.txt'),
                         bert_tokenizer='bert-base-chinese')

        self.id2tag = self.seq.tagging()
        self.checkpoint = torch.load(os.path.join(self.cws_path, "cws.pth"), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.checkpoint['net'], False)
    def cws(self,sents):
        results = []
        for sent in sents:

            test_pkg = {'token': sent, 'tag': ' '.join(['0'] * len(sent))}
            tokens, tags, mask = self.seq.transformer.item2id(test_pkg)
            tokens = torch.LongTensor([tokens])
            result = ''

            with torch.no_grad():
                out = self.model.predict(3, tokens, mask)
                out = out.numpy().tolist()
                out_etts = [get_word(line, self.id2tag) for line in out]

                for seg in out_etts[0]:
                    result += sent[seg['begin']:seg['end'] + 1] + ' '
                reuslt = result[:-1]
            results.append(reuslt)
        return results

