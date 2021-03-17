import os
import torch
import torch.utils.data as Data
from torch import nn
from ink.nn.models.bertlinear import BertLinear
from ink.data import load
from ..preProcessor.bertPosTagging import Dataset, classlist
from ..preProcessor.bertPosTagging.apply_text_norm import process_sent
from ..preProcessor.bertPosTagging.evaluate_funcs import format_output


class PosTagging:
    def __init__(self):
        self.prepare_model()

    def prepare_model(self):
        self.model = nn.DataParallel(BertLinear(classlist))
        path = load('pos')
        checkpoint = torch.load(os.path.join(path, "params.ckpt"), map_location=lambda storage, loc: storage)
        self.model.module.load_state_dict(checkpoint)
        self.model.eval()

    def infer_step(self, batch):
        x, at, y = batch
        if torch.cuda.is_available():
            x, at, y = x.cuda(), at.cuda(), y.cuda()
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
        """
        Args:
            sents: list[str]

        Return:
            list[list]
        """
        sents = [process_sent(' '.join(sent)).split(' ') for sent in sents]
        examples = [[sent, [0 for i in range(len(sent))]] for sent in sents]
        dataset = Dataset(examples=examples)
        res = self.infer_epoch(Data.DataLoader(dataset, batch_size=4, num_workers=4))
        return res
