import torch.nn as nn
from transformers import AutoModel, AutoConfig
from transformers import BertTokenizer as TransformerBertTokenizer


class BertEncoder(nn.Module):
    def __init__(self,
                 bert_route='bert-base-chinese'
                 ):
        super().__init__()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(bert_route))
        self.out_dim = self.encoder.config.hidden_size

    def forward(self, inputs, mask):
        # sent,tags,masks: (batch * seq_length)
        return self.encoder(inputs, mask)


class BertTokenizer:
    def __init__(self, version='bert-base-chinese'):
        self.tokenizer = TransformerBertTokenizer.from_pretrained(version)

    def __call__(self, text):
        return self.tokenizer.tokenize(text)

    def token2id(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class Bert(nn.Module):
    def __init__(self, version='bert-base-chinese'):
        super().__init__()
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained(version))

    def frozen(self, bool):
        bool = not bool
        for param in self.bert.base_model.parameters():
            param.requires_grad = bool

    def forward(self, x, output_only=True, **kwargs):
        if output_only:
            return self.bert(x, **kwargs)[0]
        return self.bert(x, **kwargs)
