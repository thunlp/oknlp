# encoding:utf-8
import os
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


def sequence_mask(sequence_length, max_len=80, device=None, padding=True):   # sequence_length :(batch_size, )
    if (not padding):
        max_len = torch.max(sequence_length)
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.to(device)
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def prepare_pack_padded_sequence(inputs_words,
                                 seq_lengths,
                                 descending=True):
    """
    :param use_cuda:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices


def initial_parameter(net, initial_method=None):

    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        # classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            init_method(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    net.apply(weights_init)


def summary(model: nn.Module):

    train = []
    nontrain = []
    buffer = []

    def layer_summary(module: nn.Module):
        def count_size(sizes):
            return reduce(lambda x, y: x * y, sizes)

        for p in module.parameters(recurse=False):
            if p.requires_grad:
                train.append(count_size(p.shape))
            else:
                nontrain.append(count_size(p.shape))
        for p in module.buffers():
            buffer.append(count_size(p))
        for subm in module.children():
            layer_summary(subm)

    layer_summary(model)
    total_train = sum(train)
    total_nontrain = sum(nontrain)
    total = total_train + total_nontrain
    strings = []
    strings.append('Total params: {:,}'.format(total))
    strings.append('Trainable params: {:,}'.format(total_train))
    strings.append('Non-trainable params: {:,}'.format(total_nontrain))
    strings.append("Buffer params: {:,}".format(sum(buffer)))
    max_len = len(max(strings, key=len))
    bar = '-' * (max_len + 3)
    strings = [bar] + strings + [bar]
    print('\n'.join(strings))
    return total, total_train, total_nontrain


def get_dropout_mask(drop_p: float, tensor: torch.Tensor):

    mask_x = torch.ones_like(tensor)
    nn.functional.dropout(mask_x, p=drop_p,
                          training=False, inplace=True)
    return mask_x


def _get_file_name_base_on_postfix(dir_path, postfix):

    files = list(filter(lambda filename: filename.endswith(postfix), os.listdir(os.path.join(dir_path))))
    if len(files) == 0:
        raise FileNotFoundError(f"There is no file endswith *{postfix} file in {dir_path}")
    elif len(files) > 1:
        raise FileExistsError(f"There are multiple *{postfix} files in {dir_path}")
    return os.path.join(dir_path, files[0])
