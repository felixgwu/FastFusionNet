# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa
#
# Modified by Felix Wu: adding RCModelProto, CnnDocReader, FusionNet, BiDAF

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers
from typing import IO, List, Iterable, Tuple
from qa.encoder import *


class RCModelProto(nn.Module):
    '''prototype of the reading comprehension models'''

    def __init__(self, opt, padding_idx=0, embedding=None):
        super().__init__()
        # Store config
        self.opt = opt
        self.setup_emb_modules(padding_idx, embedding)

    def setup_emb_modules(self, padding_idx=0, embedding=None):
        opt = self.opt
        # Word embeddings
        self.paired_input_size = opt['embedding_dim']
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding(embedding.size(0),
                                          embedding.size(1),
                                          padding_idx=padding_idx)
            self.embedding.weight.data[2:, :] = embedding[2:, :]
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial'] + 2:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        else:  # random initialized
            self.embedding = nn.Embedding(opt['vocab_size'],
                                          opt['embedding_dim'],
                                          padding_idx=padding_idx)

        # Character embeddings
        if opt['use_max_char_emb']:
            self.max_char_emb = nn.Embedding(301, opt['max_char_emb_size'], padding_idx=260)
            self.paired_input_size += opt['max_char_emb_size']

        # Contextualized embeddings
        if opt['use_cove']:
            self.CoVe = layers.MTLSTM(opt, embedding, padding_idx=padding_idx)
            self.paired_input_size += self.CoVe.output_size

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = self.paired_input_size + opt['num_features']
        question_input_size = self.paired_input_size
        if opt['use_feat_emb']:
            if opt['pos']:
                doc_input_size += opt['pos_dim']
                self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            if opt['ner']:
                doc_input_size += opt['ner_dim']
                self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
        else:
            if opt['pos']:
                doc_input_size += opt['pos_size']
            if opt['ner']:
                doc_input_size += opt['ner_size']

        # Projection for attention weighted question
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'],
                                                  dropout=opt['dropout_rnn'],
                                                  variational_dropout=opt['variational_dropout'],
                                                  )
            # self.qemb_match = layers.FullAttention(
            #     full_size=opt['embedding_dim'],
            #     hidden_size=opt['embedding_dim'],
            #     num_level=1,
            #     dropout=opt['dropout_rnn'],
            #     variational_dropout=opt['variational_dropout'],
            # )
            doc_input_size += opt['embedding_dim']
        if opt['use_demb']:
            self.demb_match = layers.SeqAttnMatch(opt['embedding_dim'],
                                                  dropout=opt['dropout_rnn'],
                                                  variational_dropout=opt['variational_dropout'],
                                                  )
            # self.demb_match = layers.FullAttention(
            #     full_size=opt['embedding_dim'],
            #     hidden_size=opt['embedding_dim'],
            #     num_level=1,
            #     dropout=opt['dropout_rnn'],
            #     variational_dropout=opt['variational_dropout'],
            # )
            question_input_size += opt['embedding_dim']

        if opt['use_qemb'] and opt['use_demb']:
            self.paired_input_size += opt['embedding_dim']

        self.doc_input_size = doc_input_size
        self.question_input_size = question_input_size

    def forward_emb(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char=None, x2_char=None):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        x1_char = document char indices        [batch * len_d * nchar]
        x2_char = question char indices        [batch * len_q * nchar]
        """

        if self.training and self.opt.get('word_dropout_c', 0.) > 0.:
            dropout_mask = torch.rand(x1.size()) < self.opt.get('word_dropout_c', 0.)
            if x1.is_cuda:
                dropout_mask = dropout_mask.cuda()
            x1.data.masked_fill_(dropout_mask, 1)
        if self.training and self.opt.get('word_dropout_q', 0.) > 0.:
            dropout_mask = torch.rand(x2.size()) < self.opt.get('word_dropout_q', 0.)
            if x1.is_cuda:
                dropout_mask = dropout_mask.cuda()
            x2.data.masked_fill_(dropout_mask, 1)


        def dropout(x, p=self.opt['dropout_rnn']):
            return layers.dropout(x, p=p,
                                  training=self.training, variational=self.opt['variational_dropout'] and x.dim() == 3)
        feat_dict = {}
        x1_all_list, x2_all_list = [], []
        x1_paired_list, x2_paired_list = [], []

        if x1_f is not None:
            x1_all_list.append(x1_f)

        # Embed both document and question
        if self.opt['use_word_emb'] or self.opt['use_qemb'] or self.opt['use_demb']:
            x1_emb = self.embedding(x1)
            x2_emb = self.embedding(x2)
            if self.opt['dropout_emb'] > 0:
                x1_emb = dropout(x1_emb, self.opt['dropout_emb'])
                x2_emb = dropout(x2_emb, self.opt['dropout_emb'])
            feat_dict['x1_emb'] = x1_emb
            feat_dict['x2_emb'] = x2_emb

        if self.opt['use_word_emb']:
            x1_all_list.append(x1_emb)
            x2_all_list.append(x2_emb)
            x1_paired_list.append(x1_emb)
            x2_paired_list.append(x2_emb)

        if self.opt['use_max_char_emb']:
            bs = x1_char.size(0)
            char_len = self.opt['max_char_emb_max_len']
            char_dim = self.opt['max_char_emb_size']
            x1_max_char_emb = self.max_char_emb(x1_char[:, :, :char_len].contiguous().view(-1, char_len)).view(bs, -1, char_len, char_dim).max(2)[0]
            x2_max_char_emb = self.max_char_emb(x2_char[:, :, :char_len].contiguous().view(-1, char_len)).view(bs, -1, char_len, char_dim).max(2)[0]
            if self.opt['dropout_emb'] > 0:
                x1_max_char_emb = dropout(x1_max_char_emb, self.opt['dropout_emb'])
                x2_max_char_emb = dropout(x2_max_char_emb, self.opt['dropout_emb'])
            x1_all_list.append(x1_max_char_emb)
            x2_all_list.append(x2_max_char_emb)
            x1_paired_list.append(x1_max_char_emb)
            x2_paired_list.append(x2_max_char_emb)

        # Contextualized embeddings
        if self.opt['use_cove']:
            _, x1_cove = self.CoVe(x1, x1_mask)
            _, x2_cove = self.CoVe(x2, x2_mask)
            if self.opt['dropout_emb'] > 0:
                x1_cove = dropout(x1_cove, self.opt['dropout_emb'])
                x2_cove = dropout(x2_cove, self.opt['dropout_emb'])
            x1_all_list.append(x1_cove)
            x2_all_list.append(x2_cove)
            x1_paired_list.append(x1_cove)
            x2_paired_list.append(x2_cove)
            feat_dict['x1_cove'] = x1_cove
            feat_dict['x2_cove'] = x2_cove

        if self.opt['use_feat_emb']:
            if self.opt['pos']:
                x1_pos_emb = self.pos_embedding(x1_pos)
                x1_all_list.append(x1_pos_emb)
                feat_dict['x1_pos_emb'] = x1_pos_emb
            if self.opt['ner']:
                x1_ner_emb = self.ner_embedding(x1_ner)
                x1_all_list.append(x1_ner_emb)
                feat_dict['x1_ner_emb'] = x1_ner_emb
        else:
            if self.opt['pos']:
                x1_all_list.append(x1_pos)
                feat_dict['x1_pos'] = x1_pos
            if self.opt['ner']:
                x1_all_list.append(x1_ner)
                feat_dict['x1_ner'] = x1_ner

        # Add attention-weighted question representation (word level fusion)
        if self.opt['use_qemb']:
            x1_qemb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            # x1_qemb = self.qemb_match(x1_emb, x2_emb, x2_emb, x2_mask)
            x1_all_list.append(x1_qemb)
            feat_dict['x1_qemb'] = x1_qemb

        if self.opt['use_demb']:
            x2_demb = self.demb_match(x2_emb, x1_emb, x1_mask)
            # x2_demb = self.demb_match(x2_emb, x1_emb, x1_emb, x1_mask)
            x2_all_list.append(x2_demb)
            feat_dict['x2_demb'] = x2_demb

        if self.opt['use_qemb'] and self.opt['use_demb']:
            x1_paired_list.append(x1_qemb)
            x2_paired_list.append(x2_demb)

        x1_paired_emb = layers.maskzero(torch.cat(x1_paired_list, 2), x1_mask.unsqueeze(-1))
        x2_paired_emb = layers.maskzero(torch.cat(x2_paired_list, 2), x2_mask.unsqueeze(-1))
        x1_full_emb = layers.maskzero(torch.cat(x1_all_list, 2), x1_mask.unsqueeze(-1))
        x2_full_emb = layers.maskzero(torch.cat(x2_all_list, 2), x2_mask.unsqueeze(-1))
        return x1_paired_emb, x2_paired_emb, x1_full_emb, x2_full_emb, feat_dict

    def forward(self, x1, x1_f, x0_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char=None, x2_char=None):
        raise NotImplementedError


class RnnDocReader(RCModelProto):
    """Network for the Document Reader module of DrQA."""

    def __init__(self, opt, padding_idx=0, embedding=None):
        super().__init__(opt, padding_idx, embedding)

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=self.doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            variational_dropout=opt['variational_dropout'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=self.question_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            variational_dropout=opt['variational_dropout'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )

        if opt['end_gru']:
            self.end_gru = nn.GRUCell(doc_hidden_size, question_hidden_size)
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char=None, x2_char=None, logit=False):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        # Embed both document and question
        x1_paired_emb, x2_paired_emb, x1_full_emb, x2_full_emb, feat_dict = self.forward_emb(x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char, x2_char)

        if self.opt['sentence_level']:
            x1_mask_backup = x1_mask
            x1_full_emb = layers.split_sentences(x1_full_emb, sent_lens)
            x1_mask = layers.split_sentences(x1_mask.unsqueeze(-1), sent_lens).select(2, 0)
            # print('after:', x1_full_emb.size())
            # print('x1_mask:', x1_mask.size())
            # print(x1_full_emb.data.type())
            # print(x1_mask.data.type())
        #     print(sent_lens)
        #     assert False

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(x1_full_emb, x1_mask)

        if self.opt['sentence_level']:
            x1_mask = x1_mask_backup
            doc_hiddens = layers.combine_sentences(doc_hiddens, sent_lens)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_full_emb, x2_mask)
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        start_logits = self.start_attn(doc_hiddens, question_hidden, x1_mask, logit=True)
        start_scores = F.log_softmax(start_logits, 1) if self.training else F.softmax(start_logits, 1)
        if self.opt['end_gru']:
            weights = start_scores.exp() if self.training else start_scores
            weighted_doc_hidden = layers.weighted_avg(doc_hiddens, weights)
            question_v_hidden = self.end_gru(weighted_doc_hidden, question_hidden)
            end_logits = self.end_attn(doc_hiddens, question_v_hidden, x1_mask, logit=True)
        else:
            end_logits = self.end_attn(doc_hiddens, question_hidden, x1_mask, logit=True)

        if logit:
            return start_logits, end_logits
        else:
            end_scores = F.log_softmax(end_logits, 1) if self.training else F.softmax(end_logits, 1)
            return start_scores, end_scores


class CnnDocReader(RCModelProto):
    def __init__(self, opt, padding_idx=0, embedding=None):
        super().__init__(opt, padding_idx, embedding)

        # RNN document encoder
        self.doc_rnn = layers.DilatedResNet(
            input_size=self.doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            dilation_base=2,
            dilation_layers=3,
            dilation_offset=1,
        )

        # RNN question encoder
        self.question_rnn = layers.DilatedResNet(
            input_size=self.question_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            dilation_base=1,
        )

        # Output sizes of rnn encoders
        doc_hidden_size = opt['hidden_size']
        question_hidden_size = opt['hidden_size']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )

        if opt['end_gru']:
            self.end_gru = nn.GRUCell(doc_hidden_size, question_hidden_size)
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char=None, x2_char=None):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        # Embed both document and question
        x1_paired_emb, x2_paired_emb, x1_full_emb, x2_full_emb, feat_dict = self.forward_emb(x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char, x2_char)

        if self.opt['sentence_level']:
            x1_mask_backup = x1_mask
            print('before:', x1_full_emb.size())
            x1_full_emb = layers.split_sentences(x1_full_emb, sent_lens)
            print('after:', x1_full_emb.size())
            print(sent_lens)
        #     assert False

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(x1_full_emb, x1_mask)

        if self.opt['sentence_level']:
            x1_mask = x1_mask_backup
            doc_hiddens = layers.combine_sentences(doc_hiddens, sent_lens)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_full_emb, x2_mask)
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask, log=self.training)
        if self.opt['end_gru']:
            weights = start_scores.exp() if self.training else start_scores
            weighted_doc_hidden = layers.weighted_avg(doc_hiddens, weights)
            question_v_hidden = self.end_gru(weighted_doc_hidden, question_hidden)
            end_scores = self.end_attn(doc_hiddens, question_v_hidden, x1_mask, log=self.training)
        else:
            end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask, log=self.training)
        return start_scores, end_scores


class FusionNet(RCModelProto):
    """Network for FusionNet."""

    def __init__(self, opt, padding_idx=0, embedding=None):
        super().__init__(opt, padding_idx, embedding)

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=self.doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=2,
            dropout_rate=opt['dropout_rnn'],
            # dropout_output=opt['dropout_rnn_output'],
            variational_dropout=opt['variational_dropout'],
            concat_layers=True,
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
            residual=opt['residual'],
            squeeze_excitation=opt['squeeze_excitation'],
        )

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=self.question_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=2,
            dropout_rate=opt['dropout_rnn'],
            # dropout_output=opt['dropout_rnn_output'],
            variational_dropout=opt['variational_dropout'],
            concat_layers=True,
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
            residual=opt['residual'],
            squeeze_excitation=opt['squeeze_excitation'],
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * 2 * opt['hidden_size']
        question_hidden_size = doc_hidden_size

        self.question_urnn = layers.StackedBRNN(
            input_size=question_hidden_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['fusion_understanding_layers'],
            dropout_rate=opt['dropout_rnn'],
            variational_dropout=opt['variational_dropout'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
            residual=opt['residual'],
            squeeze_excitation=opt['squeeze_excitation'],
            concat_layers=False,
        )

        self.multi_level_fusion = layers.FullAttention(
            full_size=self.paired_input_size + doc_hidden_size,
            hidden_size=2 * 3 * opt['hidden_size'],
            num_level=3,
            dropout=opt['dropout_rnn'],
            variational_dropout=opt['variational_dropout'],
        )

        self.doc_urnn = layers.StackedBRNN(
            input_size=2 * 5 * opt['hidden_size'],
            hidden_size=opt['hidden_size'],
            num_layers=opt['fusion_understanding_layers'],
            dropout_rate=opt['dropout_rnn'],
            variational_dropout=opt['variational_dropout'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
            residual=opt['residual'],
            squeeze_excitation=opt['squeeze_excitation'],
            concat_layers=False,
        )


        self.self_boost_fusions = nn.ModuleList()
        self.doc_final_rnns = nn.ModuleList()
        full_size=self.paired_input_size + 4 * 3 * opt['hidden_size']
        for i in range(self.opt['fusion_self_boost_times']):
            self.self_boost_fusions.append(layers.FullAttention(
                full_size=full_size,
                hidden_size=2 * opt['hidden_size'],
                num_level=1,
                dropout=opt['dropout_rnn'],
                variational_dropout=opt['variational_dropout'],
            ))

            self.doc_final_rnns.append(layers.StackedBRNN(
                input_size=4 * opt['hidden_size'],
                hidden_size=opt['hidden_size'],
                num_layers=opt['fusion_final_layers'],
                dropout_rate=opt['dropout_rnn'],
                variational_dropout=opt['variational_dropout'],
                rnn_type=opt['rnn_type'],
                padding=opt['rnn_padding'],
                residual=opt['residual'],
                squeeze_excitation=opt['squeeze_excitation'],
                concat_layers=False,
            ))
            full_size += 2 * opt['hidden_size']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.quesiton_merge_attns = nn.ModuleList()

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(2 * opt['hidden_size'])

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            2 * opt['hidden_size'],
            2 * opt['hidden_size'],
        )

        if opt['end_gru']:
            self.end_gru = nn.GRUCell(2 * opt['hidden_size'], 2 * opt['hidden_size'])

        self.end_attn = layers.BilinearSeqAttn(
            2 * opt['hidden_size'],
            2 * opt['hidden_size'],
        )

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char=None, x2_char=None, logit=False):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        def dropout(x, p=self.opt['dropout_rnn']):
            return layers.dropout(x, p=p,
                                  training=self.training, variational=self.opt['variational_dropout'] and x.dim() == 3)

        # Embed both document and question
        x1_paired_emb, x2_paired_emb, x1_full_emb, x2_full_emb, feat_dict = self.forward_emb(x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char, x2_char)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(x1_full_emb, x1_mask)
        # Encode question with RNN
        question_hiddens = self.question_rnn(x2_full_emb, x2_mask)

        # Question Understanding
        question_u_hiddens = self.question_urnn(question_hiddens, x2_mask)

        # Fully-Aware Multi-level Fusion
        doc_HoW = torch.cat([x1_paired_emb, doc_hiddens], 2)
        question_HoW = torch.cat([x2_paired_emb, question_hiddens], 2)
        question_cat_hiddens = torch.cat([question_hiddens, question_u_hiddens], 2)
        doc_fusions = self.multi_level_fusion(doc_HoW, question_HoW, question_cat_hiddens, x2_mask)

        # Document Understanding
        doc_u_hiddens = self.doc_urnn(torch.cat([doc_hiddens, doc_fusions], 2), x1_mask)

        # Fully-Aware Self-Boosted Fusion
        self_boost_HoW = torch.cat([x1_paired_emb, doc_hiddens, doc_fusions, doc_u_hiddens], 2)

        for i in range(len(self.self_boost_fusions)):
            doc_self_fusions = self.self_boost_fusions[i](self_boost_HoW, self_boost_HoW, doc_u_hiddens, x1_mask)
            
            # Final document representation
            doc_final_hiddens = self.doc_final_rnns[i](torch.cat([doc_u_hiddens, doc_self_fusions], 2), x1_mask)
            if i < len(self.self_boost_fusions) - 1:
                self_boost_HoW = torch.cat([self_boost_HoW, doc_final_hiddens], 2)
                doc_u_hiddens = doc_final_hiddens

        # Encode question with RNN + merge hidden, 2s
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_u_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(dropout(question_u_hiddens), x2_mask)
        question_u_hidden = layers.weighted_avg(question_u_hiddens, q_merge_weights)

        # Predict start and end positions
        start_logits = self.start_attn(dropout(doc_final_hiddens), dropout(question_u_hidden), x1_mask, logit=True)
        if self.opt['sentence_level']:
            start_logits = layers.combine_sentences(start_logits, sent_lens)

        start_scores = F.log_softmax(start_logits, 1) if self.training else F.softmax(start_logits, 1)
        if self.opt['end_gru']:
            weights = start_scores.exp() if self.training else start_scores
            weighted_doc_hidden = layers.weighted_avg(doc_final_hiddens, weights)
            question_v_hidden = self.end_gru(dropout(weighted_doc_hidden), dropout(question_u_hidden))
            # question_v_hidden = layers.dropout(question_v_hidden)
            end_logits = self.end_attn(dropout(doc_final_hiddens), dropout(question_v_hidden), x1_mask, logit=True)
        else:
            end_logits = self.end_attn(doc_final_hiddens, question_u_hidden, x1_mask, logit=True)

        if self.opt['sentence_level']:
            end_logits = layers.combine_sentences(end_logits, sent_lens)

        if logit:
            return start_logits, end_logits
        else:
            end_scores = F.log_softmax(end_logits, 1) if self.training else F.softmax(end_logits, 1)
            return start_scores, end_scores


class BiDAF(RCModelProto):
    """simple BiDAF model (without char-cnn)"""

    def __init__(self, opt, padding_idx=0, embedding=None):
        super().__init__(opt, padding_idx, embedding)
        # Store config

        # RNN document encoder
        self.doc_enc = layers.StackedBRNN(
            input_size=self.doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=1,
            dropout_rate=opt['dropout_rnn'],
            variational_dropout=opt['variational_dropout'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
        )

        # RNN question encoder
        if self.doc_input_size == self.question_input_size:
            self.question_enc = self.doc_enc
        else:
            self.question_enc = layers.StackedBRNN(
                input_size=self.question_input_size,
                hidden_size=opt['hidden_size'],
                num_layers=1,
                dropout_rate=opt['dropout_rnn'],
                variational_dropout=opt['variational_dropout'],
                rnn_type=opt['rnn_type'],
                padding=opt['rnn_padding'],
            )

        # Context-Query Attention Layer
        self.biattn = layers.BiAttn(2*opt['hidden_size'])

        # Model Encoder Layer
        self.model_enc = layers.StackedBRNN(
            input_size=opt['hidden_size']*8,
            hidden_size=opt['hidden_size'],
            num_layers=2,
            dropout_rate=opt['dropout_rnn'],
            variational_dropout=opt['variational_dropout'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
        )

        self.end_enc = layers.StackedBRNN(
            input_size=opt['hidden_size']*14,
            hidden_size=opt['hidden_size'],
            num_layers=1,
            dropout_rate=opt['dropout_rnn'],
            variational_dropout=opt['variational_dropout'],
            rnn_type=opt['rnn_type'],
            padding=opt['rnn_padding'],
        )

        # Bilinear attention for span start/end
        self.start_attn = layers.LinearSeqAttn(10 * opt['hidden_size'])
        self.end_attn = layers.LinearSeqAttn(10 * opt['hidden_size'])

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char=None, x2_char=None):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """

        batch_size = x1.size(0)
        x1_len = x1.size(1)
        x2_len = x2.size(1)

        def dropout(x, p=self.opt['dropout_rnn']):
            return layers.dropout(x, p=p,
                                  training=self.training, variational=self.opt['variational_dropout'] and x.dim() == 3)

        # Embed both document and question
        x1_paired_emb, x2_paired_emb, x1_full_emb, x2_full_emb, feat_dict = self.forward_emb(x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, sent_lens, x1_char, x2_char)

        # Encode document with RNN
        doc_hiddens = self.doc_enc(x1_full_emb, x1_mask)
        # Encode question with RNN
        question_hiddens = self.question_enc(x2_full_emb, x2_mask)

        # Context-Query Attention
        outputs = self.biattn(doc_hiddens, question_hiddens, x1_mask, x2_mask)
        outputs[1] = outputs[1].expand_as(outputs[2]) # Q2C is a vector need to expand 
        outputs.append(doc_hiddens)
        p0 = torch.cat(outputs, 2) 

        # Predict start and end positions
        g1 = self.model_enc(p0, x1_mask)
        start_scores = self.start_attn(dropout(torch.cat([g1, p0], 2)), x1_mask, log=self.training)
        alpha = start_scores.exp() if self.training else start_scores
        a1i = layers.weighted_avg(g1, alpha).unsqueeze_(1)
        g2 = self.end_enc(torch.cat([p0, g1, a1i.expand_as(g1), g1 * a1i], 2), x1_mask) 
        end_scores = self.end_attn(dropout(torch.cat([g2, p0], 2)), x1_mask, log=self.training)
        return start_scores, end_scores
