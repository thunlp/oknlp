# encoding:utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .model_utils import prepare_pack_padded_sequence


def mask2len(mask):
    seq_lengths = torch.sum(mask, dim=1)
    return seq_lengths


class BILSTM(nn.Module):
    def __init__(self,
                 hidden_size=200,
                 num_layer=1,
                 input_size=128,
                 dropout_p=0.5,
                 bi_tag=True):

        super().__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layer,
                            batch_first=True,
                            dropout=dropout_p if num_layer > 1 else 0,
                            bidirectional=bi_tag)

    def forward(self, inputs, mask):
        seq_lengths = mask2len(mask)
        inputs, seq_lengths, desorted_indice = prepare_pack_padded_sequence(inputs, seq_lengths)
        embeddings_packed = pack_padded_sequence(inputs, seq_lengths, batch_first=True)
        output, (h, c) = self.lstm(embeddings_packed)

        hidden = torch.cat((h[0, :, :], h[1, :, :]), dim=-1)
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = output[desorted_indice]
        output = F.dropout(output, p=self.dropout_p, training=self.training)
        output = torch.tanh(output)

        hidden = hidden[desorted_indice]
        hidden = F.dropout(hidden, p=self.dropout_p, training=self.training)
        hidden = torch.tanh(hidden)

        return output, hidden
