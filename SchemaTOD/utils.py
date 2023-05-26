import torch
import numpy as np
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence


def pad(seqs, pad_token_id, trunc_len=1024):
    if isinstance(seqs[0][0], int):
        seqs = [torch.tensor(seq[-trunc_len:]) for seq in seqs]
        seqs = pad_sequence(seqs, batch_first=True, padding_value=pad_token_id)

    elif isinstance(seqs[0][0][0], int):
        seqs = [[torch.tensor(i[-trunc_len:]) for i in seq] for seq in seqs]
        max_len = max([len(i) for seq in seqs for i in seq])
        pad_seqs = []
        for seq in seqs:
            pad_tensor_list = [F.pad(i, (0, max_len - len(i)), value=pad_token_id) for i in seq]
            pad_seqs.append(torch.stack(pad_tensor_list))
        seqs = pad_sequence(pad_seqs, batch_first=True, padding_value=pad_token_id)

    else:
        seqs = [[[torch.tensor(j[-trunc_len:]) for j in i] for i in seq] for seq in seqs]
        max_right_len = max([len(j) for seq in seqs for i in seq for j in i])
        max_down_len = max([len(i) for seq in seqs for i in seq])
        tmp_seqs = []
        for seq in seqs:
            pad_seqs = []
            for i in seq:
                pad_tensor_list = [F.pad(j, (0, max_right_len - len(j)), value=pad_token_id) for j in i]
                pad_seqs.append(torch.stack(pad_tensor_list))
            pad_tensor_list = [F.pad(i, (0, 0, 0, max_down_len - len(i)), value=pad_token_id) for i in pad_seqs]
            tmp_seqs.append(torch.stack(pad_tensor_list))
        seqs = pad_sequence(tmp_seqs, batch_first=True, padding_value=pad_token_id)

    return seqs


def pad_seqs_gpt(sequences, pad_id, maxlen=None):
    lengths = [len(x) for x in sequences]

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)

    maxlen = seq_maxlen if seq_maxlen <= 1024 else 1024

    # tokenizer.encode('<|endoftext|>') = ['50256']
    # All labels set to ``-100`` are ignored (masked), the loss is only
    # computed for labels in ``[0, ..., config.vocab_size]`` (from modeling_gpt2.GPT2LMHeadModel)

    x = (np.ones((num_samples, maxlen)) * pad_id)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list was found in padSeqs')
        # trunc method = 'pre'
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc)

        # pad method = 'post'
        x[idx, :len(trunc)] = trunc

    return x, [len(i) for i in x]


def pad_left(seqs, pad_token_id):
    seq_maxlen = np.max([len(x) for x in seqs])
    pad_seqs = (np.ones((len(seqs), seq_maxlen)) * pad_token_id)
    for idx, seq in enumerate(seqs):
        pad_seqs[idx, -len(seq):] = np.asarray(seq)
    return pad_seqs.tolist()


def to_tensor(seqs):
    for k in seqs:
        if isinstance(seqs[k], list):
            seqs[k] = torch.tensor(seqs[k]).long()
    return seqs


def to_device(seqs, device='cpu'):
    for k in seqs:
        if isinstance(seqs[k], torch.Tensor):
            seqs[k] = seqs[k].to(device)
    return seqs


def flatten_list(two_d_list):
    return [j for i in two_d_list for j in i]
