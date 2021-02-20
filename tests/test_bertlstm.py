import numpy as np
import torch
import unittest
from ink.nn.models.bertlstmcrf import BERT_LSTM


class TestBertLSTM(unittest.TestCase):
    def test_predict(self):
        # only test input.shape[0] == output.shape[0] (which is batch_size)
        model = BERT_LSTM(input_size=300, hidden_size=200, label_sizes=[8], toplayer='CRF')
        batch_size = 10
        tokens = torch.from_numpy(np.random.randint(0, 100, size=(batch_size, 80)).astype(np.int64))
        mask = torch.from_numpy(np.random.randint(0, 2, size=(batch_size, 80)).astype(np.int64)) < 1
        out = model.predict(8, tokens, mask)
        self.assertEqual(out.shape[0], batch_size)
