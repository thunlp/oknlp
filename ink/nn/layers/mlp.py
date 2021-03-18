import torch
import torch.nn as nn

from .model_utils import initial_parameter


class MultiDecoder(nn.Module):
    def __init__(self, hidden, sizegroup, exp=False):
        super(MultiDecoder, self).__init__()
        self.mlp = {}
        for size in sizegroup:
            ln = torch.nn.Linear(hidden, size + (2 if exp else 0))  # MLP([hidden,size+(2 if exp else 0)])
            nn.init.xavier_uniform_(ln.weight)
            self.mlp[size] = ln

    def forward(self, x, outsz):
        return self.mlp[outsz](x)

    def to(self, device):
        super().to(device)
        for size in self.mlp.keys():
            self.mlp[size] = self.mlp[size].to(device)


class MLP(nn.Module):

    def __init__(self, size_layer, activation='relu', output_activation=None, initial_method=None, dropout=0.5):

        super(MLP, self).__init__()
        self.hiddens = nn.ModuleList()
        self.output = None
        self.output_activation = output_activation
        for i in range(1, len(size_layer)):
            if i + 1 == len(size_layer):
                self.output = nn.Linear(size_layer[i - 1], size_layer[i])
            else:
                self.hiddens.append(nn.Linear(size_layer[i - 1], size_layer[i]))

        self.dropout = nn.Dropout(p=dropout)

        actives = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        if not isinstance(activation, list):
            activation = [activation] * (len(size_layer) - 2)
        elif len(activation) == len(size_layer) - 2:
            pass
        else:
            raise ValueError(
                f"the length of activation function list except {len(size_layer) - 2} but got {len(activation)}!")
        self.hidden_active = []
        for func in activation:
            if callable(activation):
                self.hidden_active.append(activation)
            elif func.lower() in actives:
                self.hidden_active.append(actives[func])
            else:
                raise ValueError("should set activation correctly: {}".format(activation))
        if self.output_activation is not None:
            if callable(self.output_activation):
                pass
            elif self.output_activation.lower() in actives:
                self.output_activation = actives[self.output_activation]
            else:
                raise ValueError("should set activation correctly: {}".format(activation))
        initial_parameter(self, initial_method)

    def forward(self, x):

        for layer, func in zip(self.hiddens, self.hidden_active):
            x = self.dropout(func(layer(x)))
        x = self.output(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        x = self.dropout(x)
        return x
