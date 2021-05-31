import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertLinearSigmoid(nn.Module):
    def __init__(self, len_tokenizer):
        super().__init__()

        self.bert = AutoModel.from_config(AutoConfig.from_pretrained("bert-base-multilingual-cased"))
        self.bert.resize_token_embeddings(len_tokenizer)
        self.predict = nn.Linear(768, 130)
        self.sig = nn.Sigmoid()

        for x in self.bert.parameters():
            x.requires_grad = False

    def forward(self, x, pos, mask_tensor=None):
        x = self.bert(x, mask_tensor)
        pos = pos.reshape(-1, 1, 1).repeat(1, 1, x.size(2))
        x = torch.gather(x['last_hidden_state'], 1 , pos).squeeze(dim=1)
        x = self.sig(self.predict(x))
        return x
