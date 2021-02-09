import torch
import os
from ..preProcessor import SeqDataset
from ink.nn.models.bertlstmcrf import BERT_LSTM

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


def cws(sents):
    cws_path = ''
    basic_path = ''
    model = BERT_LSTM(input_size = 300, hidden_size = 200, label_sizes = [3], toplayer = 'CRF')
    seq  = SeqDataset(file_path = os.path.join(cws_path , "cws.txt"), embedding_path = os.path.join(basic_path , 'sgns300'),
                     vocab_path = os.path.join(basic_path , 'vocab.pkl'),tag_path = os.path.join(cws_path , 'tagset_cws.txt'),
                     bert_tokenizer = 'bert-base-chinese')
    results = []
    id2tag = seq.tagging()
    checkpoint = torch.load(os.path.join(cws_path , "cws.pth"), map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['net'],False)
    for sent in sents:

        test_pkg = {'token': sent, 'tag': ' '.join(['0'] * len(sent))}
        tokens, tags, mask = seq.transformer.item2id(test_pkg)
        tokens = torch.LongTensor([tokens])
        tags = torch.LongTensor([tags])
        result = ''

        with torch.no_grad():
            out, loss, tag = model(3, tokens, mask, tags)
            out, tag = out.numpy().tolist(), tag.numpy().tolist()
            out_etts = [get_word(line, id2tag) for line in out]

            for seg in out_etts[0]:
                result += sent[seg['begin']:seg['end']+1]+' '
            reuslt = result[:-1]
        results.append(reuslt)
    return results

