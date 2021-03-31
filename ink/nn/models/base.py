import torch
from torch import nn
import torch.nn.functional as F
from ..layers.pretrain import Backbone


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = Backbone()
        self.dropout = nn.Dropout(0.1)
        self.num_class = 0
        self.cls = nn.Linear(768, self.num_class)

    # def froze_bert(self, bool):
    #     self.bert.frozen(bool)

    def expand_to(self, new_class):
        old_weight = self.cls.weight.data
        old_bias = self.cls.bias.data
        self.cls = nn.Linear(768, new_class)
        self.cls.weight.data[:self.num_class] = old_weight
        self.cls.bias.data[:self.num_class] = old_bias
        self.num_class = new_class
        self.lossfunc = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([1 if i == 0 else 1 for i in range(new_class)]).cuda())
    
    def forward(self, sent, attention_mask, y=None, hidden_output=False, logit_output=False):
        h = self.bert(sent, output_only=True, attention_mask=attention_mask)
        x = self.dropout(h)
        x = self.cls(x)
        res = []
        if y == None: # infer
            if hidden_output or logit_output:
                res.append([])
                if hidden_output:
                    res[-1].append(h)
                if logit_output:
                    res[-1].append(x)
            else:
                res.append(torch.argmax(F.softmax(x, dim=-1), dim=-1))
        else: # train
            res.append(self.lossfunc(x.transpose(1, 2), y))
            if hidden_output or logit_output:
                res.append([])
            if hidden_output:
                res[-1].append(h)
            if logit_output:
                res[-1].append(x)
        return res if len(res)>1 else res[0]
