import torch
from transformers import BertTokenizer
from ..BaseCWS import BaseCWS
from ....utils.dataset import Dataset
from ....nn.models import BertSeq as Model
import torch.utils.data as Data
from functools import reduce
from ....utils.format_output import format_output
from ....utils.process_io import merge_result, split_text_list
from ....data import load
from ....config import config
import os

labels = reduce(lambda x, y: x + y, [[f"{kd}-{l}" for kd in ('B', 'I', 'O')] for l in ('SEG',)])


class BertCWS(BaseCWS):
    """使用Bert模型实现的CWS算法
    """
    def __init__(self, device=None):
        self.cws_path = load('cws')
        self.model = Model()
        self.model.expand_to(len(labels), device)
        if device is None:
            device = config.default_device
        self.model.load_state_dict(
            torch.load(os.path.join(self.cws_path, "cws_bert.ckpt"), map_location=torch.device(device)))
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        super().__init__(device)

    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)

    def __call__(self, sents):
        self.sents, is_end_list = split_text_list(sents, 126)
        self.test_dataset = Dataset(self.sents, self.tokenizer)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=8, num_workers=0)
        return merge_result(self.infer_epoch(self.test_loader), is_end_list)

    def infer_step(self, batch):
        x, y, at = batch
        x = x.to(self.device)
        y = y.to(self.device)
        at = at.to(self.device)
        with torch.no_grad():
            p = self.model(x, at)
            p = p.to(self.device)
            mask = y != -1
        return torch.where(mask, p, -1).cpu().tolist(), mask.cpu().tolist()

    def infer_epoch(self, infer_loader):
        pred, mask = [], []
        for batch in infer_loader:
            p, m = self.infer_step(batch)
            pred += p
            mask += m
        results = []
        formatted_output = format_output(self.sents, pred, labels + ['O'])
        for sent, out in zip(self.sents, formatted_output):
            results.append([sent[j[1]:j[2] + 1] for j in out])
        return results
