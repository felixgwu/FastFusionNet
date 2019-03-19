# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa
#
# Modified by Felix Wu
# Modification:
#   - change the logger name
#   - save & load optimizer state dict
#   - change the dimension of inputs (for POS and NER features)

import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from .utils import AverageMeter, EMA
from .rnn_reader import *


logger = logging.getLogger(__name__)


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict else 0
        self.train_loss = AverageMeter()

        # Building network.
        if opt['model_type'] == 'drqa':
            self.network = RnnDocReader(opt, embedding=embedding)
        elif opt['model_type'] == 'gldr-drqa':
            self.network = CnnDocReader(opt, embedding=embedding)
        elif opt['model_type'] == 'fusionnet':
            self.network = FusionNet(opt, embedding=embedding)
        elif opt['model_type'] == 'bidaf':
            self.network = BiDAF(opt, embedding=embedding)
        else:
            print('UNKNOWN model_type: ' + opt['model_type'])
            raise NotImplementedError
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters, opt['learning_rate'],
                                          weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters, opt['learning_rate'],
                                        betas=(opt['beta1'], opt['beta2']),
                                        weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        print(self.optimizer)

        if opt['ema_decay'] < 1.:
            print('using EMA')
            self.ema = EMA(opt['ema_decay'])
            for name, param in self.network.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)

        num_params = sum(p.data.numel() for p in parameters
            if p.data.data_ptr() != self.network.embedding.weight.data.data_ptr())
        print("{} parameters".format(num_params))

    def update(self, ex):
        # Train mode
        self.network.train()

        # Transfer to GPU
        with torch.no_grad():
            if self.opt['cuda']:
                inputs = [e.cuda(non_blocking=True) if torch.is_tensor(e) else e for e in ex[:10]]
                target_s, target_e = ex[10].cuda(non_blocking=True), ex[11].cuda(non_blocking=True)
            else:
                inputs = ex[:10]
                target_s, target_e = ex[10], ex[11]

        # Run forward
        score_s, score_e = self.network(*inputs)

        # Compute loss and accuracies
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        self.train_loss.update(loss.item(), ex[0].size(0))

        # warm_start
        if self.opt['warm_start'] and self.updates <= 1000:
            lr = self.opt['learning_rate'] / math.log(1002.) * math.log(self.updates + 2)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if self.opt['grad_clipping'] > 0.:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                           self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Exponential Moving Average
        if hasattr(self, 'ema'):
            for name, param in self.network.named_parameters():
               if param.requires_grad:
                   param.data = self.ema(name, param.data)

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        with torch.no_grad():
            # Transfer to GPU
            if next(self.network.parameters()).is_cuda:
                inputs = [e.cuda(non_blocking=True) if torch.is_tensor(e) else e for e in ex[:10]]
            else:
                inputs = ex[:10]

            # Run forward
            score_s, score_e = self.network(*inputs)
            if type(score_s) is list:
                score_s, score_e = score_s[-1], score_e[-1]


            # Transfer to CPU/normal tensors for numpy ops
            score_s = score_s.data.cpu()
            score_e = score_e.data.cpu()

            # Get argmax text spans
            text = ex[-2]
            spans = ex[-1]
            predictions = []
            max_len = self.opt['max_len'] or score_s.size(1)
            for i in range(score_s.size(0)):
                scores = torch.ger(score_s[i], score_e[i])
                scores.triu_().tril_(max_len - 1)
                scores = scores.numpy()
                s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
                s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                predictions.append(text[i][s_offset:e_offset])

            return predictions

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename, epoch, best_val_score=0.):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch,
            'best_val_score': best_val_score,
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
        return self

    def cpu(self):
        self.network.cpu()
        return self
