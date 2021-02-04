import torch.nn as nn
from transformers import BertModel


class Bert_Encoder(nn.Module):
    def __init__(self,
                 bert_route='bert-base-chinese'
                 ):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_route)
        self.out_dim = self.encoder.config.hidden_size

    def forward(self, inputs, mask):
        # sent,tags,masks: (batch * seq_length)
        return self.encoder(inputs, mask)
