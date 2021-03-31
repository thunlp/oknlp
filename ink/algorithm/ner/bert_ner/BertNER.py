import torch
from ..BaseNER import BaseNER
from ....utils.dataset import Dataset
from ....nn.models import Model
from torch import nn
import torch.utils.data as Data
from functools import reduce
from ....utils.format_output import format_output
from ....data import load
import os

labels = ['O'] + reduce(lambda x,y:x+y, [[f"{kd}-{l}" for kd in ('B','I', 'O')] for l in ('PER','LOC','ORG')])

class Tester(BaseNER):
    def __init__(self, sents):
        self.ner_path = load('ner')
        self.sents = sents
        self.prepare_dataset()
        self.prepare_model()
    def prepare_model(self):
        self.model = Model()
        self.model.expand_to(len(labels))
        self.model.load_state_dict(
            torch.load(os.path.join(self.ner_path,"ner_bert.ckpt")))
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
        return torch.where(mask, p, 0).cpu().tolist(), mask.cpu().tolist()

    def infer_epoch(self, infer_loader):
        pred, mask = [], []
        for batch in infer_loader:
            p, m = self.infer_step(batch)
            pred += p
            mask += m
        results =[]
        for i in format_output(self.sents, pred,labels):
            res = []
            for j in i:
                tmp ={}
                tmp['type'] = j[0]
                tmp['begin'] =j[1]
                tmp['end'] = j[2]
                res.append(tmp)
            results.append(res)
        return results

    def __call__(self):
        return self.infer_epoch(self.test_loader)

