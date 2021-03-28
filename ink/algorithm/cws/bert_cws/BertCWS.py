import torch
import os
from ..BaseCWS import BaseCWS
from ....utils.seq_dataloader import SeqDataset
from ....nn.models import BertLSTMCRF
from ....data import load


class BertCWS(BaseCWS):
    def __init__(self, device=None):
        self.cws_path = load('cws')
        self.model = BertLSTMCRF(input_size=300, hidden_size=200, label_sizes=[3], toplayer='CRF')
        self.seq = SeqDataset(
            tag_path=os.path.join(self.cws_path, 'tagset_cws.txt'),
            bert_tokenizer='bert-base-chinese')
        self.id2tag = self.seq.tagging()
        self.checkpoint = torch.load(os.path.join(self.cws_path, "cws.pth"), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(self.checkpoint['net'], False)
        self.model.eval()

        super().__init__(device)

    def to(self, device: str):
        self.model = self.model.to(device)
        return super().to(device)

    def __call__(self, sents):
        results = []
        for sent in sents:
            test_pkg = {'token': sent, 'tag': ' '.join(['0'] * len(sent))}
            tokens, tags, mask = self.seq.transformer.item2id(test_pkg)
            tokens = torch.LongTensor([tokens])
            result = []
            with torch.no_grad():
                out = self.model.predict(3, tokens.to(self.device), mask.to(self.device))
                out = out.cpu().tolist()
                out_etts = [self.get_word(line, self.id2tag) for line in out]
                for seg in out_etts[0]:
                    result.append(sent[seg['begin']:seg['end'] + 1])
            results.append(result)
        return results

    def get_word(self, path, tag_map):
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
