import torch
from ..BaseNER import BaseNER
from ....utils.dataset import Dataset
from ....nn.models import BertSeq as Model
import torch.utils.data as Data
from functools import reduce
from transformers import BertTokenizer
from ....utils.format_output import format_output
from ....data import load
from ....config import config
import os

labels = ['O'] + reduce(lambda x, y: x + y, [[f"{kd}-{l}" for kd in ('B', 'I', 'O')] for l in ('PER', 'LOC', 'ORG')])


class BertNER(BaseNER):
    """使用Bert模型实现的NER算法
    """
    def __init__(self, device=None):
        self.ner_path = load('ner')
        self.model = Model()
        self.model.expand_to(len(labels), device)
        if device is None:
            device = config.default_device
        self.model.load_state_dict(
            torch.load(os.path.join(self.ner_path, "ner_bert.ckpt"), map_location=torch.device(device)))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        super().__init__(device)

    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)

    def __call__(self, sents):
        self.sents = sents
        self.test_dataset = Dataset(self.sents, self.tokenizer)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=8, num_workers=0)
        return self.infer_epoch(self.test_loader)

    def infer_step(self, batch):
        x, y, at = batch
        x = x.to(self.device)
        y = y.to(self.device)
        at = at.to(self.device)
        with torch.no_grad():
            p = self.model(x, at)
            p = p.to(self.device)
            mask = y != -1
        return torch.where(mask, p, 0).cpu().tolist(), mask.cpu().tolist()

    def infer_epoch(self, infer_loader):
        pred, mask = [], []
        for batch in infer_loader:
            p, m = self.infer_step(batch)
            pred += p
            mask += m
        results = []
        for i in format_output(self.sents, pred, labels):
            res = []
            for j in i:
                tmp = {}
                tmp['type'] = j[0]
                tmp['begin'] = j[1]
                tmp['end'] = j[2]
                res.append(tmp)
            results.append(res)
        return results
