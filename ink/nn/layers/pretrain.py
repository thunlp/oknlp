import torch.nn as nn
from transformers import  BertModel as md


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = md.from_pretrained("bert-base-chinese").train() # default is eval()
    
    def frozen(self, bool):
        bool = not bool
        for param in self.bert.base_model.parameters():
            param.requires_grad = bool
    
    def forward(self, x, output_only=True, **kwargs):
        if output_only:
            return self.bert(x, **kwargs)[0]
        return self.bert(x, **kwargs)
