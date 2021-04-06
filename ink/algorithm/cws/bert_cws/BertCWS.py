import torch
from ..BaseCWS import BaseCWS
from ....utils.dataset import Dataset
from ....nn.models import BertSeq as Model
from torch import nn
import torch.utils.data as Data
from functools import reduce
from ....utils.format_output import format_output
from ....data import load
import os

labels = reduce(lambda x,y:x+y, [[f"{kd}-{l}" for kd in ('B','I','O')] for l in ('SEG',)])

class BertCWS(BaseCWS):
    def __init__(self, device=None):
        self.cws_path = load('cws')
        self.model = Model()
        self.model.expand_to(len(labels))
        self.model.load_state_dict(
            torch.load(os.path.join(self.cws_path,"cws_bert.ckpt")))
        self.model.eval()
        super().__init__(device)

    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)

    def __call__(self,sents):
        self.sents = sents
        self.test_dataset = Dataset(self.sents)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=4, num_workers=0)
        return self.infer_epoch(self.test_loader)    

    def infer_step(self, batch):
        x, y, at = batch
        x = x.to(self.device)
        y = y.to(self.device)
        at = at.to(self.device)
        with torch.no_grad():
            p = self.model(x, at)
            mask = y != -1
        return torch.where(mask, p, -1).cpu().tolist(), mask.cpu().tolist()

    def infer_epoch(self, infer_loader):
        pred, mask = [], []
        for batch in infer_loader:
            p, m = self.infer_step(batch)
            pred += p
            mask += m
        results =[]
        for i in range(len(self.sents)):
            tmp = format_output(self.sents, pred, labels + ['O'])[i]
            results.append([self.sents[i][j[1]:j[2]+1] for j in tmp])
        return results


