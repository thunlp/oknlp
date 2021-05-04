import torch
from ..BaseNER import BaseNER
from ....utils.dataset import Dataset
from ....nn.models import BertSeq as Model
import torch.utils.data as Data
from functools import reduce
from transformers import BertTokenizer
from ....utils.format_output import format_output
from ....utils.process_io import split_text_list, merge_result
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
        self.sents, is_end_list = split_text_list(sents, 126)
        self.test_dataset = Dataset(self.sents, self.tokenizer)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=8, num_workers=0)
        split_ans_list = self.infer_epoch(self.test_loader)
        count = 0
        for i, sent in enumerate(self.sents):
            split_ans = split_ans_list[i]
            for d in split_ans:
                d['begin'] += count
                d['end'] += count
            count += len(sent)
            if is_end_list[i]:
                count = 0
        return merge_result(split_ans_list, is_end_list)

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
            results.append([{'type': j[0], 'begin': j[1], 'end': j[2]} for j in i])
        return results
