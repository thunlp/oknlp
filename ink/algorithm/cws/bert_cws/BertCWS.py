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
    def __init__(self, sents):
        self.cws_path = load('cws')
        self.sents = sents
        self.prepare_dataset()
        self.prepare_model()
    def prepare_model(self):
        self.model = Model()
        self.model.expand_to(len(labels))
        self.model.load_state_dict(
            torch.load(os.path.join(self.cws_path,"cws_bert.ckpt")))
        self.model = nn.DataParallel(self.model.cuda())
        self.model.eval()

    def prepare_dataset(self):
        self.test_dataset = Dataset(self.sents)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=4, num_workers=4)

    def infer_step(self, batch):
        x, y, at = batch
        x = x.cuda()
        y = y.cuda()
        at = at.cuda()
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
            tmp = format_output(self.sents, pred,labels + ['O'])[i]
            results.append(' '.join([self.sents[i][j[1]:j[2]+1] for j in tmp]))

        return results

    def __call__(self):
        return self.infer_epoch(self.test_loader)
