import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BertLinearSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
         
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained("bert-base-multilingual-cased"))
        self.predict = nn.Linear(768, 130)
        self.sig = nn.Sigmoid()

        for x in self.bert.parameters():
            x.requires_grad = False

    def forward(self, x, pos):
        x = self.bert(x)
        x = x['last_hidden_state']
        x = x[0, pos, :]

        x = self.predict(x)
        x = self.sig(x)

        return x
