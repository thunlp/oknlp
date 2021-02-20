import torch
import os
from torch.autograd import Variable
from ..preProcessor import SeqDataset
from ink.nn.models.bertlstmcrf import BERT_LSTM
from ink.utils.format import get_entity
torch.manual_seed(2018)

def ner(sents):
    ner_path = ''
    basic_path = ''
    model = BERT_LSTM(input_size=300,hidden_size=200,label_sizes=[8],toplayer='CRF')
    seq =  SeqDataset(file_path = os.path.join(ner_path, "msra_ner.txt"),embedding_path = os.path.join(basic_path, 'sgns300'),
                      vocab_path = os.path.join(basic_path, 'vocab.pkl'),tag_path = os.path.join(ner_path,'tagset.txt'),
                      bert_tokenizer = 'bert-base-chinese')
    id2tag = seq.tagging()
    checkpoint = torch.load(os.path.join(ner_path,"bert2.pth"),map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['net'],False)
    result = []
    for sent in sents:
        test_pkg = {'token':sent,'tag':' '.join(['O']*len(sent))}
        tokens,tags, masks = seq.transformer.item2id(test_pkg)
        tokens = torch.LongTensor([tokens])
        tags = torch.LongTensor([tags])
        lens = torch.LongTensor([lens])
        #s = Score()
        with torch.no_grad():
            out, loss, tag = model(8, tokens, masks, tags)
            out, tag = out.numpy().tolist(), tag.numpy().tolist()
            out_etts = [get_entity(line,id2tag) for line in out]
            result.append(out_etts)
    return result
