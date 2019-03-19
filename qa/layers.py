# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# Modified by Felix Wu

import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from torch.autograd.function import InplaceFunction
from oldsru import SRUCell


# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

def variational_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training == False or p == 0:
        return x
    dropout_mask = Variable(1.0 / (1-p) * torch.bernoulli((1-p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1)), requires_grad=False)
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0, training=False, variational=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if p > 0:
        if variational and len(x.size()) == 3: # if x is (batch * len * input_size)
            return variational_dropout(x, p=p, training=training)
        else:
            return F.dropout(x, p=p, training=training)
    else:
        return x


class SizeDropout(nn.Module):
    def __init__(self, input_size, min_size, dim, rescale=True):
        super().__init__()
        self.min_size = min_size
        self.input_size = input_size
        self.dim = dim
        self.eval_size = input_size
        self.rescale = rescale
        if min_size < input_size:
            mask = torch.cat([torch.ones(min_size), torch.arange(input_size - min_size, 0, -1) / (input_size - min_size + 1)], dim=0) 
        else:
            mask = torch.ones(input_size)
        self.register_buffer('mask', torch.ones(input_size))
        # self.register_buffer('eval_mask', mask)
        self.eval_mask = mask.cuda()
        self.train_size = input_size
        self.generate_mask(1)

    def sample_train_size(self):
        if self.input_size == self.min_size:
            return self.input_size
        self.train_size = self.min_size + min(int(torch.rand(1)[0] * (self.input_size - self.min_size + 1)), self.input_size) ## take the min in case of getting 1 from torch.rand
        return self.train_size

    def generate_mask(self, max_dim):
        curr_mask = self.mask.clone()
        if self.train_size < self.input_size:
            curr_mask[self.train_size:] = 0
        for i in range(self.dim):
            curr_mask.unsqueeze_(0)
        for i in range(self.dim+1, max_dim):
            curr_mask.unsqueeze_(-1)
        self.curr_mask_var = Variable(curr_mask, requires_grad=False)

    def generate_eval_mask(self, max_dim):
        if self.rescale:
            curr_mask = self.eval_mask.clone()
        else:
            curr_mask = torch.ones(self.input_size).cuda()
        if self.eval_size < self.input_size:
            curr_mask[self.eval_size:] = 0
        for i in range(self.dim):
            curr_mask.unsqueeze_(0)
        for i in range(self.dim+1, max_dim):
            curr_mask.unsqueeze_(-1)
        self.curr_eval_mask_var = Variable(curr_mask, requires_grad=False)

    def forward(self, x, resample=True, mask=None):
        assert x.size(self.dim) == self.input_size, 'x: {}, input_size: {}'.format(x.size(), self.input_size)
        if self.input_size == self.min_size:
            return x
        if self.training:
            if resample:
                self.sample_train_size()
                self.generate_mask(x.dim())
            elif isinstance(mask, Variable):
                self.curr_mask_var = mask
            elif x.dim() != self.curr_mask_var.dim() or type(x.data) != type(self.curr_mask_var.data):
                '''# of dim doesn't match generate the mask again'''
                self.generate_mask(x.dim())
            x = x * self.curr_mask_var
        else:
            self.generate_eval_mask(x.dim())
            x = x * self.curr_eval_mask_var
        return x

    def __repr__(self):
        return '{}(input_size={}, min_size={}, dim={}, rescale={}, eval_size={})'.format(
            self.__class__.__name__, self.input_size, self.min_size, self.dim, self.rescale, self.eval_size)


class LayerNorm(nn.Module):
    '''Layer Norm implementation source: https://github.com/pytorch/pytorch/issues/1959'''
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LayerNormChannelFirst(nn.Module):
    '''Layer Norm implementation source: https://github.com/pytorch/pytorch/issues/1959'''
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features).view(1, -1, 1))
        self.beta = nn.Parameter(torch.zeros(features).view(1, -1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-2, keepdim=True)
        std = x.std(-2, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class StackedBRNN(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN, 'sru': SRUCell}
    SRU_TYPES = {'sru', 'sru-v2'}

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 variational_dropout=True,
                 residual=False,
                 squeeze_excitation=0,
                 sd_min_size=0, sd_rescale=True,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.variational_dropout = variational_dropout
        self.num_layers = num_layers
        self.residual = residual
        self.squeeze_excitation = squeeze_excitation
        self.concat_layers = concat_layers
        self.sd_min_size = sd_min_size
        self.sd_rescale = sd_rescale
        self.rnns = nn.ModuleList()
        self.rnn_type = rnn_type
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            if rnn_type in self.SRU_TYPES:
                self.rnns.append(self.RNN_TYPES[rnn_type](input_size, hidden_size,
                                                            dropout=dropout_rate,
                                                            rnn_dropout=dropout_rate,
                                                            use_tanh=1,
                                                            bidirectional=True))
            else:
                self.rnns.append(self.RNN_TYPES[rnn_type](input_size, hidden_size,
                                                          num_layers=1,
                                                          bidirectional=True))
        if sd_min_size > 0:
            self.sds = nn.ModuleList()
            for i in range(num_layers):
                self.sds.append(SizeDropout(hidden_size, sd_min_size, 3, sd_rescale))
        if squeeze_excitation > 0:
            self.ses = nn.ModuleList()
            for i in range(num_layers):
                self.ses.append(nn.Sequential(nn.Linear(hidden_size*2, hidden_size*2//self.squeeze_excitation),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(hidden_size*2//self.squeeze_excitation, hidden_size*2),
                                              nn.Sigmoid()))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        # if (self.padding or not self.training) and not self.rnn_type == 'sru':
        if self.padding and not self.rnn_type in self.SRU_TYPES:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)
        lengths_var = Variable(x_mask.data.eq(0).long().sum(1).squeeze().float().unsqueeze(1), requires_grad=False)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0 and self.rnn_type not in self.SRU_TYPES:
                rnn_input = dropout(rnn_input, p=self.dropout_rate, training=self.training,
                                    variational=self.variational_dropout)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            if self.residual and rnn_output.size() == rnn_input.size():
                rnn_output = rnn_output + outputs[-1]

            if self.sd_min_size > 0:
                bs, l, hs = rnn_output.size()
                rnn_output = self.sds[i](rnn_output.view(bs, l, 2, hs//2)).view(bs, l, hs)

            if self.squeeze_excitation > 0:
                rnn_output = rnn_output * self.ses[i](rnn_output.sum(0) / lengths_var).unsqueeze(0)
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = dropout(output, p=self.dropout_rate, training=self.training,
                             variational=self.variational_dropout)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths_var = Variable(lengths[idx_sort].float().unsqueeze(1), requires_grad=False)
        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        # rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                rnn_input = dropout(rnn_input, p=self.dropout_rate, training=self.training,
                                    variational=self.variational_dropout)
            rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths)

            # if self.dropout_rate > 0:
            #     dropout_input = F.dropout(rnn_input.data,
            #                               p=self.dropout_rate,
            #                               training=self.training)
            #     rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
            #                                             rnn_input.batch_sizes)

            rnn_output = self.rnns[i](rnn_input)[0]
            rnn_output = nn.utils.rnn.pad_packed_sequence(rnn_output)[0]

            if self.residual and rnn_output.size() == outputs[-1].size():
                rnn_output = rnn_output + outputs[-1]

            if self.sd_min_size > 0:
                bs, l, hs = rnn_output.size()
                rnn_output = self.sds[i](rnn_output.view(bs, l, 2, hs//2)).view(bs, l, hs)

            if self.squeeze_excitation > 0:
                rnn_output = rnn_output * self.ses[i](rnn_output.sum(0) / lengths_var).unsqueeze(0)
            outputs.append(rnn_output)

        # Unpack everything
        # for i, o in enumerate(outputs[1:], 1):
            # outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class DilatedResNet(nn.Module):
    """Dilated ResNet with GRU to replace BRNN."""

    def __init__(self, input_size, hidden_size, num_layers,
                 dilation_layers=1, dilation_base=1, dilation_offset=0,
                 input_padding=0, masked=True,
                 growing_mode='block', # ['block', 'layer']
                 block_type='dilated_conv', # ['dilated_conv', 'dilated_sep_conv', 'sep_conv']
                 activation_type='glu', # ['glu', 'relu']
                 dropout_rate=0, dropout_output=False):
        super(DilatedResNet, self).__init__()
        # self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.input_padding = input_padding
        # self.concat_layers = concat_layers
        if activation_type == 'glu':
            self.reduce_block = nn.Sequential(
                nn.Conv1d(input_size, hidden_size*2, 3, padding=1 + input_padding),
                nn.GLU(dim=1))
        else:
            self.reduce_block = nn.Sequential(
                nn.Conv1d(input_size, hidden_size, 3, padding=1 + input_padding),
                nn.ReLU(inplace=True))
        self.cnns = nn.ModuleList()
        self.masked = masked
        assert num_layers % 2 == 1, 'num_layers=' + str(num_layers) +' is not odd'
        for i in range(num_layers // 2):
            if block_type == 'sep_conv':
                if growing_mode == 'block':
                    kernel_size = 2 ** (i - dilation_offset + 2) - 1 if dilation_offset <= i < dilation_offset + dilation_layers else 3
                    kernel_size = (kernel_size, kernel_size)
                elif growing_mode == 'layer':
                    kernel_size = [1, 1]
                    kernel_size[0] = 2 ** (2*i+2-dilation_offset) - 1 if dilation_offset <= (2*i+1) < dilation_offset + dilation_layers else 3
                    kernel_size[1] = 2 ** (2*i+3-dilation_offset) - 1 if dilation_offset <= (2*i+2) < dilation_offset + dilation_layers else 3
                else:
                    raise NotImplementedError
                dilation = 1
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            elif block_type in {'dilated_conv', 'dilated_sep_conv'}:
                if growing_mode == 'block':
                    dilation = dilation_base ** (i - dilation_offset + 1) if dilation_offset <= i < dilation_offset + dilation_layers else 1
                elif growing_mode == 'layer':
                    dilation = [1, 1]
                    dilation[0] = dilation_base ** (2*i+1-dilation_offset) if dilation_offset <= (2*i+1) < dilation_offset + dilation_layers else 1
                    dilation[1] = dilation_base ** (2*i+2-dilation_offset) if dilation_offset <= (2*i+2) < dilation_offset + dilation_layers else 1
                else:
                    raise NotImplementedError
                padding = dilation
                kernel_size = 3
            else:
                raise NotImplementedError

            if block_type == 'dilated_conv':
                Block = GLUResBlock
            elif block_type in {'dilated_sep_conv', 'sep_conv'}:
                Block = GLUResBlock_sep
            else:
                raise NotImplementedError

            self.cnns.append(Block(hidden_size, hidden_size,
                                   hidden_size, kernel_size=kernel_size,
                                   padding=padding,
                                   dilation=dilation,
                                   dropout_rate=dropout_rate,
                                   activation_type=activation_type))

    def forward(self, x, x_mask=None):
        # swap filter dim and sequence dim
        if self.input_padding > 0 and self.masked and x_mask is not None:
            x_mask = F.pad(x_mask.unsqueeze(1).unsqueeze(2), (self.input_padding, self.input_padding, 0, 0), 'constant', True)[:, 0, 0, :]
        x = x.transpose(1, 2)
        if self.dropout_output and self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate,
                          training=self.training)
        x = self.reduce_block(x)
        for cnn in self.cnns:
            x = cnn(x, x_mask)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate,
                          training=self.training)
        x = x.transpose(1, 2)
        return x.contiguous()


class GLUResBlock(nn.Module):
    '''GLU Res Block
    input -> drop1 -> conv1 -> GLU1 -> drop2 -> conv2 -> GLU2 --> residual
    add residual back to input
    '''
    def __init__(self, input_size, hidden_size, output_size, kernel_size=3,
                 padding=1, groups=1, dilation=1, dropout_rate=0, activation_type='glu'):
        super(GLUResBlock, self).__init__()
        if type(dilation) is int:
            dilation = (dilation, dilation)
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        if type(padding) is int:
            padding = (padding, padding)

        self.dropout_rate = dropout_rate
        self.drop1 = nn.Dropout2d(dropout_rate)
        self.activation_type = activation_type
        if activation_type == 'glu':
            self.conv1 = nn.Conv1d(input_size, hidden_size*2, kernel_size[0],
                                   padding=padding[0], dilation=dilation[0])
            self.act1 = nn.GLU(dim=1)
        elif activation_type == 'relu':
            self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size[0],
                                   padding=padding[0], dilation=dilation[0])
            self.act1 = nn.ReLU(inplace=True)

        self.drop2 = nn.Dropout2d(dropout_rate)
        if activation_type == 'glu':
            self.conv2 = nn.Conv1d(hidden_size, output_size*2, kernel_size[1],
                                   padding=padding[1], dilation=dilation[1])
            self.act2 = nn.GLU(dim=1)
        elif activation_type == 'relu':
            self.conv2 = nn.Conv1d(hidden_size, output_size, kernel_size[1],
                                   padding=padding[1], dilation=dilation[1])
            self.act2 = nn.ReLU(inplace=True)

    def forward(self, x, x_mask=None, masked=True):
        res = x
        res = self.drop1(res.unsqueeze(3))[:, :, :, 0]
        res = self.act1(self.conv1(x))
        if masked and x_mask is not None:
            res.masked_fill_(x_mask.unsqueeze(1), 0)

        res = self.drop2(res.unsqueeze(3))[:, :, :, 0]
        res = self.act2(self.conv2(x))
        if masked and x_mask is not None:
            res.masked_fill_(x_mask.unsqueeze(1), 0)

        if x.size(1) == res.size(1):
            x = x + res
        elif x.size(1) > res.size(1):
            res = res + x[:, :res.size(1)]
            x = res
        else:
            x = x + res[:, :x.size(1)]
        return x


class GLUResBlock_sep(nn.Module):
    '''GLU Res Block
    input -> drop1 -> conv1 -> GLU1 -> drop2 -> conv2 -> GLU2 --> residual
    add residual back to input
    '''
    def __init__(self, input_size, hidden_size, output_size, kernel_size=3,
                 padding=1, groups=1, dilation=1, dropout_rate=0, activation_type='glu'):
        super().__init__()
        if type(dilation) is int:
            dilation = (dilation, dilation)
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        if type(padding) is int:
            padding = (padding, padding)

        self.dropout_rate = dropout_rate
        self.drop1 = nn.Dropout2d(dropout_rate)
        self.activation_type = activation_type
        if activation_type == 'glu':
            self.conv1_1 = nn.Conv1d(input_size, input_size, kernel_size[0],
                                     groups=input_size, padding=padding[0], dilation=dilation[0])
            self.conv1_2 = nn.Conv1d(input_size, hidden_size*2, 1)
            self.act1 = nn.GLU(dim=1)
        elif activation_type == 'relu':
            self.conv1_1 = nn.Conv1d(input_size, input_size, kernel_size[0],
                                     groups=input_size, padding=padding[0], dilation=dilation[0])
            self.conv1_2 = nn.Conv1d(input_size, hidden_size, 1)
            self.act1 = nn.ReLU(inplace=True)

        self.drop2 = nn.Dropout2d(dropout_rate)
        if activation_type == 'glu':
            self.conv2_1 = nn.Conv1d(hidden_size, hidden_size, kernel_size[1],
                                     groups=hidden_size, padding=padding[1], dilation=dilation[1])
            self.conv2_2 = nn.Conv1d(hidden_size, output_size*2, 1)
            self.act2 = nn.GLU(dim=1)
        elif activation_type == 'relu':
            self.conv2_1 = nn.Conv1d(hidden_size, hidden_size, kernel_size[1],
                                     groups=hidden_size, padding=padding[1], dilation=dilation[1])
            self.conv2_2 = nn.Conv1d(hidden_size, output_size, 1)
            self.act2 = nn.ReLU(inplace=True)

    def forward(self, x, x_mask=None, masked=True):
        res = x
        res = self.drop1(res.unsqueeze(3)).squeeze(3)
        res = self.act1(self.conv1_2(self.conv1_1(x)))
        if masked and x_mask is not None:
            res.masked_fill_(x_mask.unsqueeze(1), 0)

        res = self.drop2(res.unsqueeze(3)).squeeze(3)
        res = self.act2(self.conv2_2(self.conv2_1(x)))
        if masked and x_mask is not None:
            res.masked_fill_(x_mask.unsqueeze(1), 0)

        if x.size(1) == res.size(1):
            x = x + res
        elif x.size(1) > res.size(1):
            res = res + x[:, :res.size(1)]
            x = res
        else:
            x = x + res[:, :x.size(1)]
        return x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, variational_dropout=True,
                 concat_layers=False, output_act=True):
        super(MLP, self).__init__()
        self.dropout_rate = dropout_rate
        self.variational_dropout = variational_dropout
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.linears = nn.ModuleList()
        self.output_act = output_act
        for i in range(num_layers):
            input_size = input_size if i == 0 else hidden_size
            self.linears.append(nn.Linear(input_size, hidden_size))

    def forward(self, x):
        original_size = x.size()
        if len(original_size) == 3:
            x = x.view(-1, original_size[2]).contiguous()

        hiddens = []
        for i in range(self.num_layers):
            if self.dropout_rate > 0.:
                x = dropout(x, p=self.dropout_rate, training=self.training,
                            variational=self.variational_dropout)
            if i < self.num_layers - 1 or self.output_act:
                x = F.relu(self.linears[i](x), inplace=True)
            hiddens.append(x)
        
        if self.concat_layers:
            x = torch.cat(hiddens, 2)

        if len(original_size) == 3:
            x = x.view(original_size[0], original_size[1], -1).contiguous()
        return x


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, hidden_size=None, identity=False, dropout=0., variational_dropout=False):
        super(SeqAttnMatch, self).__init__()
        self.dropout = dropout
        self.variational_dropout = variational_dropout
        if hidden_size is None:
            hidden_size = input_size
        if not identity:
            self.linear = nn.Linear(input_size, hidden_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask, scores_hook=None):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        if y.size(0) == 1 and x.size(0) > 1:
            y = y.repeat(x.size(0), 1, 1)
            y_mask = y_mask.repeat(x.size(0), 1)
        elif x.size(0) == 1 and y.size(0) > 1:
            x = x.repeat(y.size(0), 1, 1)
        # Project vectors
        if self.linear is not None:
            batch_size = x.size(0)
            len1 = x.size(1)
            len2 = y.size(1)
            x = dropout(x, p=self.dropout, training=self.training, variational=self.variational_dropout)
            y = dropout(y, p=self.dropout, training=self.training, variational=self.variational_dropout)
            x_proj = self.linear(x.view(-1, x.size(2))).view(batch_size, len1, -1)
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(batch_size, len2, -1)
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        if scores_hook is not None:
            scores = scores_hook(scores)

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask, log=False, logit=False):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        if y.size(0) == 1 and x.size(0) > 1:
            y = y.repeat(x.size(0), 1)
        elif x.size(0) == 1 and y.size(0) > 1:
            x = x.repeat(y.size(0), 1, 1)
            x_mask = x_mask.repeat(y.size(0), 1)

        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if logit:
            return xWy
        elif log:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy, dim=1)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy, dim=1)
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask, log=False):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        if log:
            alpha = F.log_softmax(scores, dim=1)
        else:
            alpha = F.softmax(scores, dim=1)
        return alpha


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 variational_dropout=True, aux_size=0):
        super(RNNEncoder, self).__init__()
        self.variational_dropout = variational_dropout
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size_ = (input_size + 2 * hidden_size * i)
            if i == 0: input_size_ += aux_size
            self.rnns.append(rnn_type(input_size_, hidden_size, num_layers=1, bidirectional=True))

    def forward(self, x, x_mask, aux_input=None):
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)
        if aux_input is not None:
            aux_input = aux_input.transpose(0, 1)

        # Encode all layers
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = torch.cat(hiddens, 2)
            if i == 0 and aux_input is not None:
                rnn_input = torch.cat([rnn_input, aux_input], 2)

            # Apply dropout to input
            if self.dropout_rate > 0:
                rnn_input = dropout(rnn_input, p=self.dropout_rate, training=self.training,
                                    variational=self.variational_dropout)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            hiddens.append(rnn_output)

        # Transpose back
        hiddens = [h.transpose(0, 1) for h in hiddens]
        return hiddens[1:]


class MTLSTM(nn.Module):
    def __init__(self, opt, embedding=None, padding_idx=0, with_emb=True):
        """Initialize an MTLSTM

        Arguments:
            embedding (Float Tensor): If not None, initialize embedding matrix with specified embedding vectors
        """
        super(MTLSTM, self).__init__()

        self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_dim'], padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data = embedding

        state_dict = torch.load(opt['MTLSTM_path'], map_location=lambda storage, loc: storage)
        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True)

        state_dict1 = dict([(name, param.data) if isinstance(param, nn.Parameter) else (name, param)
                        for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([(name.replace('1', '0'), param.data) if isinstance(param, nn.Parameter) else (name.replace('1', '0'), param)
                        for name, param in state_dict.items() if '1' in name])
        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)

        for p in self.embedding.parameters():
            p.requires_grad = False
        for p in self.rnn1.parameters():
            p.requires_grad = False
        for p in self.rnn2.parameters():
            p.requires_grad = False

        self.output_size = 600

    def setup_eval_embed(self, eval_embed, padding_idx=0):
        """Allow evaluation vocabulary size to be greater than training vocabulary size

        Arguments:
            eval_embed (Float Tensor): Initialize eval_embed to be the specified embedding vectors
        """
        self.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx = padding_idx)
        self.eval_embed.weight.data = eval_embed

        for p in self.eval_embed.parameters():
            p.requires_grad = False

    def forward(self, x_idx, x_mask):
        """A pretrained MT-LSTM (McCann et. al. 2017).
        This LSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.

        Arguments:
            x_idx (Long Tensor): a Long Tensor of size (batch * len).
            x_mask (Byte Tensor): a Byte Tensor of mask for the input tensor (batch * len).
        """
        # emb = self.embedding if self.training else self.eval_embed
        emb = self.embedding
        x_hiddens = emb(x_idx)

        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        lens, indices = torch.sort(lengths, 0, True)

        output1, _ = self.rnn1(nn.utils.rnn.pack_padded_sequence(x_hiddens[indices], lens.tolist(), batch_first=True))
        output2, _ = self.rnn2(output1)

        output1 = nn.utils.rnn.pad_packed_sequence(output1, batch_first=True)[0]
        output2 = nn.utils.rnn.pad_packed_sequence(output2, batch_first=True)[0]

        _, _indices = torch.sort(indices, 0)
        output1 = output1[_indices]
        output2 = output2[_indices]

        return output1, output2


# Attention layer
class FullAttention(nn.Module):
    def __init__(self, full_size, hidden_size, num_level, dropout=0., variational_dropout=True):
        super(FullAttention, self).__init__()
        assert(hidden_size % num_level == 0)
        self.full_size = full_size
        self.hidden_size = hidden_size
        self.attsize_per_lvl = hidden_size // num_level
        self.num_level = num_level
        self.linear = nn.Linear(full_size, hidden_size, bias=False)
        self.linear_final = nn.Parameter(torch.ones(1, hidden_size), requires_grad = True)
        self.output_size = hidden_size
        self.dropout = dropout
        self.variational_dropout = variational_dropout
        # print("Full Attention: (atten. {} -> {}, take {}) x {}".format(self.full_size, self.attsize_per_lvl, hidden_size // num_level, self.num_level))

    def forward(self, x1_att, x2_att, x2, x2_mask):
        """
        x1_att: batch * len1 * full_size
        x2_att: batch * len2 * full_size
        x2: batch * len2 * hidden_size
        x2_mask: batch * len2
        """
        if x1_att.size(0) == 1 and x2_att.size(0) > 1:
            x1_att = x1_att.repeat(x2_att.size(0), 1, 1)
        elif x2_att.size(0) == 1 and x1_att.size(0) > 1:
            x2_att = x2_att.repeat(x1_att.size(0), 1, 1)
            x2 = x2.repeat(x1_att.size(0), 1, 1)
            x2_mask = x2_mask.repeat(x1_att.size(0), 1)

        batch_size = x1_att.size(0)
        len1 = x1_att.size(1)
        len2 = x2_att.size(1)

        x1_att = dropout(x1_att, p=self.dropout, training=self.training, variational=self.variational_dropout)
        x2_att = dropout(x2_att, p=self.dropout, training=self.training, variational=self.variational_dropout)

        x1_key = F.relu(self.linear(x1_att.view(-1, self.full_size)))
        x2_key = F.relu(self.linear(x2_att.view(-1, self.full_size)))
        final_v = self.linear_final.expand_as(x2_key)
        x2_key = final_v * x2_key

        x1_rep = x1_key.view(-1, len1, self.num_level, self.attsize_per_lvl).transpose(1, 2).contiguous().view(-1, len1, self.attsize_per_lvl)
        x2_rep = x2_key.view(-1, len2, self.num_level, self.attsize_per_lvl).transpose(1, 2).contiguous().view(-1, len2, self.attsize_per_lvl)

        scores = x1_rep.bmm(x2_rep.transpose(1, 2)).view(-1, self.num_level, len1, len2) # batch * num_level * len1 * len2

        x2_mask = x2_mask.unsqueeze(1).unsqueeze(2).expand_as(scores)
        scores.data.masked_fill_(x2_mask.data, -float('inf'))

        alpha_flat = F.softmax(scores.view(-1, len2), dim=1)
        alpha = alpha_flat.view(-1, len1, len2)
        # alpha = F.softmax(scores, dim=2)

        size_per_level = self.hidden_size // self.num_level
        atten_seq = alpha.bmm(x2.contiguous().view(-1, x2.size(1), self.num_level, size_per_level).transpose(1, 2).contiguous().view(-1, x2.size(1), size_per_level))

        return atten_seq.view(-1, self.num_level, len1, size_per_level).transpose(1, 2).contiguous().view(-1, len1, self.hidden_size)
    
    def __repr__(self):
        return "FullAttention: (atten. {} -> {}, take {}) x {}".format(self.full_size, self.attsize_per_lvl, self.hidden_size // self.num_level, self.num_level)


# For summarizing a set of vectors into a single vector
class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x = dropout(x, p=my_dropout_p, training=self.training)

        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha


class BiAttn(nn.Module):
    """ Bi-Directonal Attention from https://arxiv.org/abs/1611.01603 """
    def __init__(self, input_size, q2c: bool=True, query_dots: bool=True):
        super(BiAttn, self).__init__()
        self.input_size = input_size
        self.q2c = q2c
        self.query_dots = query_dots
        self.w_x = nn.Parameter(torch.Tensor(input_size, 1))
        self.w_y = nn.Parameter(torch.Tensor(input_size, 1))
        self.w_dot = nn.Parameter(torch.Tensor(input_size, 1))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform(self.w_x.data)
        nn.init.kaiming_uniform(self.w_y.data)
        nn.init.kaiming_uniform(self.w_dot.data)
        self.bias.data.zero_()

    def forward(self, x, y, x_mask=None, y_mask=None, raw_score_only=False):
        """
        Args:
            x: batch * len1 * hdim (context)
            y: batch * len2 * hdim (query)
            x_mask: batch * len1 (1 for padding, 0 for true)
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
        if raw_score_only:
            scores: batch * len1 * len2
        else:
            matched_seq: batch * len1 * hdim

        """
        batch_size = x.size(0)
        len1 = x.size(1)
        len2 = y.size(1)

        # get the scores
        x_ext = x.unsqueeze(2)
        y_ext = y.unsqueeze(1)
        try:
            xy = x_ext * y_ext
        except:
            print('x_ext:', x_ext.size())
            print('y_ext:', y_ext.size())
            import time
            time.sleep(10)
        
        scores = self.bias.view(1, 1, 1) + \
            x.contiguous().view(-1, self.input_size).mm(self.w_x).view(batch_size, len1, 1) + \
            y.contiguous().view(-1, self.input_size).mm(self.w_y).view(batch_size, 1, len2) + \
            xy.view(-1, self.input_size).mm(self.w_dot).view(batch_size, len1, len2)


        # fill the padding part with -inf
        if x_mask is not None:
            scores = maskneginf(scores, x_mask.unsqueeze(2))
        if y_mask is not None:
            scores = maskneginf(scores, y_mask.unsqueeze(1))

        if raw_score_only:
            return scores


        # context-to-query
        alpha = F.softmax(scores, dim=2)
        # replacing NaN with zeros (Softmax is numerically unstable)
        # no ideas how to avoid it yet
        alpha.data[alpha.data != alpha.data] = 0.

        c2q_attn = alpha.bmm(y)
        if x_mask is not None:
            c2q_attn = maskzero(c2q_attn, x_mask.unsqueeze(2))
        outputs = [c2q_attn]

        # query-to-context
        if self.q2c:
            beta = F.softmax(scores.max(2)[0], dim=1)
            q2c_attn = beta.unsqueeze(1).bmm(x)
            outputs.append(q2c_attn)

        if self.query_dots:
            outputs.append(x * c2q_attn)

        return outputs

    def __repr__(self):
        return '{}(input_size={}, q2c={}, query_dots={})'.format(
            self.__class__.__name__, self.input_size, self.q2c, self.query_dots)


class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:

            # assert attn_mask.size() == attn.size(), \
            #         'Attention mask shape {} mismatch ' \
            #         'with Attention logit tensor shape ' \
            #         '{}.'.format(attn_mask.size(), attn.size())
            # print(attn.size(), attn_mask.size())

            attn_mask = attn_mask.unsqueeze(1).expand_as(attn)
            attn.data.masked_fill_(attn_mask.data, -float('inf'))

        attn = F.softmax(attn, dim=-1)
        attn.data[attn.data != attn.data] = 0.
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class Highway(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout_rate=0.):
        super(Highway, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.reduction = (input_size != hidden_size)
        self.dropout_rate = dropout_rate

        if self.input_size != self.hidden_size:
            self.reduction = nn.Linear(input_size, hidden_size)

        self.highway_layers = nn.ModuleList()
        for i in range(num_layers):
            self.highway_layers.append(nn.Linear(hidden_size, hidden_size*2))

    def forward(self, x, x_mask=None):
        ndim = x.dim()
        if ndim == 3:
            batch_size = x.size(0)
            x_len = x.size(1)
            x = x.view(-1, x.size(2))

        if self.input_size != self.hidden_size:
            x = self.reduction(x)

        for layer in self.highway_layers:
            x_trans = layer(F.dropout(x, self.dropout_rate, training=self.training))
            gate = F.sigmoid(x_trans[:, self.hidden_size:])
            x_trans = F.relu(x_trans[:, :self.hidden_size])
            x = x * (1 - gate) + x_trans * gate

        if ndim == 3:
            x = x.view(batch_size, x_len, -1)

        return x


class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super().forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    pass


# borrowed from https://github.com/jadore801120/attention-is-all-you-need-pytorch.git
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout=0.0, input_layer_norm=False):
        super(MultiHeadAttention, self).__init__()

        if input_layer_norm:
            self.layer_norm = LayerNorm(d_model)

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_k, attn_dropout=attn_dropout)
        # if n_head * d_v != d_model:
        self.proj = BottleLinear(n_head*d_v, d_model)


        # self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, attn_mask=None):
        '''only supports self-attn'''
        if hasattr(self, 'layer_norm'):
            q = self.layer_norm(q)

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head


        mb_size, len_q, d_model = q.size()
        # mb_size, len_k, d_model = k.size()
        # mb_size, len_v, d_model = v.size()
        len_k = len_q
        len_v = len_q

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        # k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        # v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model
        k_s = q_s
        v_s = q_s

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1)
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) 

        # project back to residual size
        if hasattr(self, 'proj'):
            outputs = self.proj(outputs)
        # outputs = self.dropout(outputs)

        return outputs 


class GBEncoderBlock(nn.Module):
    '''Encoder of the Google Brain paper (QANet or AdamsNet)'''
    # TODO: dropout, layer dropout
    def __init__(self, hidden_size=128, kernel_size=7, num_layers=4, dropout_rate=0., variational_dropout=True, depth_drop=0., depth_drop_start=0, depth_drop_end=None, add_pos=True):
        '''assuming input_size == hidden_size'''
        super(GBEncoderBlock, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.variational_dropout = variational_dropout
        self.depth_drop = depth_drop
        self.depth_drop_start = depth_drop_start
        self.depth_drop_end = num_layers if depth_drop_end is None else depth_drop_end

        self.cnns = nn.ModuleList()
        for i in range(num_layers):
            # no activation?
            self.cnns.append(nn.Sequential(
                LayerNormChannelFirst(hidden_size),
                nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2, groups=hidden_size),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.ReLU(True)
            )) 
        self.self_attn = MultiHeadAttention(8, hidden_size, hidden_size//8, hidden_size//8, input_layer_norm=True)

        self.ffn = nn.Sequential(
            LayerNorm(hidden_size),
            BottleLinear(hidden_size, hidden_size*4),
            nn.ReLU(True),
            BottleLinear(hidden_size*4, hidden_size),
        )
        # add position embeding to the first block
        if add_pos:
            self.set_pos_emb(2000)

    def set_pos_emb(self, l):
        self.pos_emb = nn.Parameter(get_position_encoding(self.hidden_size, [l]).unsqueeze_(0))
        self.pos_emb.requires_grad = True
        

    def forward(self, x, x_mask=None):
        """
        TODO add x_mask
        x = batch * len * hidden_size
        """
        batch_size = x.size(0)
        x_len = x.size(1)

        drop_i = self.depth_drop_start

        if hasattr(self, 'pos_emb'):
            if x_len > self.pos_emb.size(1):
                self.set_pos_emb(x_len + 200)
            x = x + self.pos_emb[:, :x_len, :]
        # if x_mask is not None:
            # print('u1:', x.data[0].sum(1))
            # maskzero(x, x_mask.unsqueeze(2))
            # print('u2:', x.data[0].sum(1))


        x = x.transpose(1, 2)
        for cnn in self.cnns:
            drop_i += 1
            depth_drop_prob = self.depth_drop * drop_i / self.depth_drop_end
            if self.depth_drop <= 0. or torch.rand(1)[0] > depth_drop_prob:
                x_drop = dropout(x.transpose(1,2), p=self.dropout_rate, training=self.training,
                                 variational=self.variational_dropout).transpose(1, 2)
                residual = cnn(x_drop)
                if self.training and self.depth_drop > 0.:
                    residual = residual / (1 - depth_drop_prob)
                x = x + residual
            # if x_mask is not None:
            #     maskzero(x, x_mask.unsqueeze(1))
        x = x.transpose(1, 2)

        # print('t1:', x.data.sum())
        drop_i += 1
        depth_drop_prob = self.depth_drop * drop_i / self.depth_drop_end
        if self.depth_drop <= 0. or torch.rand(1)[0] > depth_drop_prob:
            x_drop = dropout(x, p=self.dropout_rate, training=self.training,
                             variational=self.variational_dropout)
            residual = self.self_attn(x_drop, x_mask)
            if self.training and self.depth_drop > 0.:
                residual = residual / (1 - depth_drop_prob)
            x = x + residual
        # print('t2:', x.data.sum())

        drop_i += 1
        depth_drop_prob = self.depth_drop * drop_i / self.depth_drop_end
        if self.depth_drop <= 0. or torch.rand(1)[0] > depth_drop_prob:
            x_drop = dropout(x, p=self.dropout_rate, training=self.training,
                             variational=self.variational_dropout)
            residual = self.ffn(x_drop)
            if self.training and self.depth_drop > 0.:
                residual = residual / (1 - depth_drop_prob)
            x = x + residual
        # if x_mask is not None:
        #     maskzero(x, x_mask.unsqueeze(2))
        return x


def get_position_encoding(emb_size, lengths, min_timescale=1.0, max_timescale=1.0e4): 
    '''
    create position embeding of size len1 (x len2 x len3 ...) x emb_size
    reference: https://github.com/tensorflow/tensor2tensor/blob/8bdecbe434d93cb1e79c0489df20fee2d5a37dc2/tensor2tensor/layers/common_attention.py#L503
    '''
    num_dims = len(lengths)
    num_timescales = emb_size // (num_dims * 2) 
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))
    inv_timescales = min_timescale * (torch.arange(num_timescales) * -log_timescale_increment).exp()
    inv_timescales.unsqueeze_(0)
    x = None
    for dim, length in enumerate(lengths):
        position = torch.arange(length).unsqueeze_(1)
        scaled_time = position * inv_timescales
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        for _ in range(dim):
            signal.unsqueeze_(0)
        for _ in range(num_dims - 1 - dim):
            signal.unsqueeze_(-2)
        x = signal if x is None else x + signal
    return x



# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


class MaskNegInf(InplaceFunction):
    @staticmethod
    def forward(ctx, input, mask=None):
        ctx.save_for_backward(mask)
        if mask is not None:
            input.masked_fill_(mask.expand_as(input), -float('inf'))
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_variables[0]
        if mask is not None:
            grad_output.masked_fill_(mask.expand_as(grad_output), 0)
        return grad_output, None


class MaskZero(InplaceFunction):
    @staticmethod
    def forward(ctx, input, mask=None):
        ctx.save_for_backward(mask)
        if mask is not None:
            input.masked_fill_(mask.expand_as(input), 0)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        print('go:', grad_output.sum())
        mask = ctx.saved_variables[0]
        if mask is not None:
            grad_output.masked_fill_(mask.expand_as(grad_output), 0)
        return grad_output, None


def maskneginf(input, mask):
    return MaskNegInf.apply(input, mask)


def maskzero(input, mask):
    return MaskZero.apply(input, mask)

def split_sentences(x, sentence_lens):
    assert x.size(0) == len(sentence_lens)
    ndim = x.dim()
    if ndim == 2:
        x = x.unsqueeze(-1)


    x = x.transpose(1, 2)
    sentences = []
    max_sentence_len = max(l for s in sentence_lens for l in s)
    for i, lens in enumerate(sentence_lens):
        pos = 0
        for l in lens:
            sentences.append(F.pad(x[i, :, pos:pos+l], (0, max_sentence_len - l)).transpose(0, 1))
            pos += l

    if ndim == 2:
        return torch.stack(sentences, 0).squeeze_(-1)
    else:
        return torch.stack(sentences, 0)

def combine_sentences(x, sentence_lens):
    ndim = x.dim()
    if ndim == 2:
        x = x.unsqueeze(-1)
    docs = []
    max_doc_len = max(sum(s) for s in sentence_lens)
    sent_id = 0
    zeros = Variable(x.data.new(max_doc_len, x.size(2)).zero_(), requires_grad=False)
    for i, lens in enumerate(sentence_lens):
        doc = []
        doc_len = sum(lens)
        for l in lens:
            doc.append(x[sent_id, :l])
            sent_id += 1
        if doc_len < max_doc_len:
            doc.append(zeros[:max_doc_len-doc_len])
        doc = torch.cat(doc, 0) 
        docs.append(doc)

    if ndim == 2:
        return torch.stack(docs, 0).squeeze(-1)
    else:
        return torch.stack(docs, 0)


def duplicate_for_sentences(x, sentence_lens):
    if not isinstance(x, Variable):
        x = Variable(x)
    assert x.size(0) == len(sentence_lens)
    ndim = x.dim()
    if ndim == 2:
        x = x.unsqueeze(-1)
    duplicated = []
    for i, lens in enumerate(sentence_lens):
        duplicated.append(x[i:i+1].repeat(len(lens), 1, 1))

    if ndim == 2:
        return torch.cat(duplicated, 0).squeeze_(-1)
    else:
        return torch.cat(duplicated, 0)

def reduce_for_sentences(x, sentence_lens):
    ndim = x.dim()
    if ndim == 2:
        x = x.unsqueeze(-1)
    reduced = []
    offset = 0
    for i, lens in enumerate(sentence_lens):
        reduced.append(x[offset])
        offset += len(lens)

    if ndim == 2:
        return torch.stack(reduced, 0).squeeze_(-1)
    else:
        return torch.stack(reduced, 0)

def replace_nan_grad_hook(grad):
    grad.data.masked_fill_(grad.data != grad.data, 0)
    return grad

def print_hook(name):
    def hook(grad):
        print('{}: {}/{}'.format(name, (grad.data != grad.data).sum(), grad.data.numel()))
    return hook


# https://github.com/pytorch/pytorch/issues/2591
def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    # x = my_where(
    #     (xm == float('inf')) | (xm == float('-inf')), 
    #     xm,
    #     xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    # return x if keepdim else x.squeeze(dim)
    output = xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True))
    return output if keepdim else output.squeeze(dim)


# https://github.com/pytorch/pytorch/issues/2591 implementation of torch.where in pytorch v3
def my_where(cond, xt, xf):
    ret = torch.zeros_like(xt)
    ret[cond] = xt[cond]
    ret[cond ^ 1] = xf[cond ^ 1]
    return ret


