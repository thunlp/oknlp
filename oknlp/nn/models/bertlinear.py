import torch
from torch import nn
import torch.nn.functional as F
from ..layers.bert import Bert


class BertLinear(nn.Module):
    def __init__(self, classlist):
        super().__init__()
        self.bert = Bert()
        total = 768  # output_features = 768
        self.cls = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(total, len(classlist)),
        )

    def froze_bert(self, bool):
        self.bert.frozen(bool)

    def forward(self, sent, attention_mask, y=None):
        x = self.bert(sent, output_only=True, attention_mask=attention_mask)
        x = self.cls(x)
        if y is None:  # infer
            return torch.argmax(F.softmax(x, dim=-1), dim=-1)
        else:  # train
            return F.cross_entropy(x.transpose(1, 2), y, ignore_index=-1)
