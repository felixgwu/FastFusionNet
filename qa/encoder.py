import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers
from typing import IO, List, Iterable, Tuple


class RnnEncoder(nn.Module):
    """Network for the Document Reader module of DrQA."""

    def __init__(self, opt):
        super().__init__()
        self.encoder_input_dim = opt.get('encoder_input_dim', 512)
        self.rnn_output_size = 2 * opt['hidden_size'] * opt['doc_layers'] if opt['concat_rnn_layers'] else 2 * opt['hidden_size']
        self.proj_size = 600 if opt['target_type'] == 'cove' else 2048
        self.rnn = layers.StackedBRNN(
            input_size=self.encoder_input_dim,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            variational_dropout=opt['variational_dropout'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
        )
        if self.proj_size != self.rnn_output_size:
            self.proj = nn.Linear(self.rnn_output_size, self.proj_size)

    def forward(self, x_emb, mask):
        bs, l = x_emb.size()[:2]
        pad_mask = (mask == 0)
        outputs = self.rnn(x_emb, pad_mask)
        if self.proj_size != self.rnn_output_size:
            outputs = self.proj(outputs.contiguous().view(bs * l, -1))
        outputs = outputs.contiguous().view(bs, l, -1)
        mask = mask.unsqueeze(-1)
        outputs = outputs * mask.float()
        return outputs
