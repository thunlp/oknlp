import os
import torch
import torch.utils.data as Data
from torch import nn
from ...nn.models import BertLinear
from ...data import load
from ...config import config
from ..preProcessor.bertPosTagging import Dataset, classlist
from ..preProcessor.bertPosTagging.apply_text_norm import process_sent
from ..preProcessor.bertPosTagging.evaluate_funcs import format_output


class PosTagging:
    def __init__(self, device=None):
        self.model = BertLinear(classlist)
        path = load('pos')
        checkpoint = torch.load(os.path.join(path, "params.ckpt"), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        if device is None:
            device = config.default_device
        self.to(device)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def infer_step(self, batch):
        x, at, y = batch
        x, at, y = x.to(self.device), at.to(self.device), y.to(self.device)
        with torch.no_grad():
            p = self.model(x, at)
            return p.cpu().tolist(), (y != -1).long().cpu().tolist()

    def infer_epoch(self, infer_loader):
        pred, mask = [], []
        for batch in infer_loader:
            p, m = self.infer_step(batch)
            pred += p
            mask += m
        return format_output(pred, mask, classlist, dims=2)

    def __call__(self, sents):
        sents = [process_sent(' '.join(sent)).split(' ') for sent in sents]
        examples = [[sent, [0 for i in range(len(sent))]] for sent in sents]
        dataset = Dataset(examples=examples)
        res = self.infer_epoch(Data.DataLoader(dataset, batch_size=4, num_workers=0))
        return res
