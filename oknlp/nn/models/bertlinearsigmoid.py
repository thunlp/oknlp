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
        n = len(pos)

        x = self.bert(x, attention_mask=mask_tensor)
        x = x['last_hidden_state']

        cat_list = []
        for i in range(n):
            cat_list.append(x[i, pos[i]:pos[i] + 1, :])
        x = torch.cat(cat_list, 0)

        x = self.predict(x)
        x = self.sig(x)

        return x
