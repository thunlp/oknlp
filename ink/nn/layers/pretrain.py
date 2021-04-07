import torch.nn as nn
from transformers import AutoConfig, AutoModel


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained("bert-base-chinese"))
    
    def frozen(self, bool):
        bool = not bool
        for param in self.bert.base_model.parameters():
            param.requires_grad = bool
    
    def forward(self, x, output_only=True, **kwargs):
        if output_only:
            return self.bert(x, **kwargs)[0]
        return self.bert(x, **kwargs)
