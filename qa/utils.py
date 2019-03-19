# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

# Modified by Felix Wu

from typing import Union, List, Dict, Any, Callable
import os
import re
import sys
import argparse
import torch
import string
import numpy as np
import unicodedata
import random

# from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from spacy.tokens import Doc
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from . import layers

from torch.autograd import Variable

nlp = spacy.load('en')
STOP_WORDS = nlp.Defaults.stop_words

# vocab_tag = [''] + list(nlp.tagger.labels)
# vocab_ent = ['', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
            #  'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
            #  'QUANTITY', 'ORDINAL', 'CARDINAL']
# inv_vocab_tag = {w: i for i, w in enumerate(vocab_tag)}
# inv_vocab_ent = {w: i for i, w in enumerate(vocab_ent)}
# Modification: remove unused functions and imports, add a boolean parser.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# General logging utilities.
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class EMA(object):
    '''exponential moving average from https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4'''

    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone().cuda()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


def load_data(opt, log=None):
    if opt['debug']:
        data_path = 'data/squad/sample-100{}.pth'
    else:
        data_path = 'data/squad/data{}.pth'
    data_path = data_path.format('' if opt['data_suffix'] == '' else '-' + opt['data_suffix'])
    if log:
        log.info('loading data from {}'.format(data_path))
    saved = torch.load(data_path)
    if log:
        log.info('done')
    # with open('data/squad/meta.msgpack', 'rb') as f:
        # meta = msgpack.load(f, encoding='utf9')
    meta = saved['meta']
    embedding = torch.Tensor(meta['embedding'])
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    if not opt['fix_embeddings']:
        embedding[1] = torch.normal(torch.zeros(opt['embedding_dim']), 1.)
    opt['pos_size'] = len(meta['vocab_tag']) if opt['use_feat_emb'] else 0
    opt['ner_size'] = len(meta['vocab_ent']) if opt['use_feat_emb'] else 0
    # global id2w, w2id, full_vocab, w2cids
    id2w = meta['vocab']
    w2id = {w:i for i, w in enumerate(id2w)}
    # with open(args.data_file, 'rb') as f:
    #     data = msgpack.load(f, encoding='utf8')
    data = saved['data']
    if 'full_vocab' in meta:
        full_vocab = meta['full_vocab']
    else:
        if log:
            log.info('getting full_vocab')
        full_vocab = set(d[6][s:e] for d in data['train'] + data['dev'] for s, e in d[7]) | set(meta['vocab'])
        saved['meta']['full_vocab'] = full_vocab
        torch.save(saved, data_path)
    if opt['valid_size'] > 0:
        perm_idx = np.random.RandomState(4444).permutation(len(data['train']))
        train = [data['train'][i] for i in perm_idx[:-opt['valid_size']]]
        dev = [data['train'][i] for i in perm_idx[-opt['valid_size']:]]
    else:
        train = data['train']
        dev = data['dev']
    if log:
        log.info('sorting dev')
    dev.sort(key=lambda x: len(x[1]), reverse=True)
    # if log:
    #     log.info('getting w2cids')
    # w2cids = {w: torch.IntTensor(ELMoCharacterMapper.convert_word_to_char_ids(w)) for w in full_vocab}
    # meta['w2cids'] = w2cids
    
    train_y = [x[-2] for x in train]
    if opt['valid_size'] > 0:
        dev_y = [x[-2] for x in dev]
    else:
        dev_y = [x[-1] for x in dev]
    if log:
        log.info('generating cids')
    train = [d[:9] + (None, None) + d[9:] for d in train]
    dev = [d[:9] + (None, None) + d[9:] for d in dev]
    if log:
        log.info('done')
    return train, dev, train_y, dev_y, embedding, opt, meta


def build_embedding(embed_file, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0 # <PAD> should be all 0 (using broadcast)

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb


def normalize_text(text):
    return unicodedata.normalize('NFD', text)


class BatchGen(object):
    def __init__(self, opt={}, data=[], batch_size=32, gpu=False, max_len=0, evaluation=False, with_cids=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.opt = {'tf': True, 'use_feat_emb': True, 'pos_size': 12, 'ner_size': 8, 'use_elmo': False}
        self.opt.update(opt)

        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu
        self.max_len = max_len
        self.with_cids = with_cids

        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        if max_len > 0:
            data = [d for d in data if len(d[1]) <= max_len]
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))

            with torch.no_grad():
                context_len = max(len(x) for x in batch[1])
                # print('context_len:', context_len)
                context_id = torch.LongTensor(batch_size, context_len).fill_(0)
                for i, doc in enumerate(batch[1]):
                    context_id[i, :len(doc)] = torch.LongTensor(doc)

                feature_len = len(batch[2][0][0])

                context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
                for i, doc in enumerate(batch[2]):
                    for j, feature in enumerate(doc):
                        context_feature[i, j, :] = torch.Tensor(feature)
                if not self.opt['tf']:
                    if self.opt['match']:
                        context_feature = context_feature[:, :, :3]
                    else:
                        context_feature = None
                else:
                    if not self.opt['match']:
                        context_feature = context_feature[:, :, 3:]

                if self.opt['use_feat_emb']:
                    context_tag = torch.LongTensor(batch_size, context_len).fill_(0)
                    for i, doc in enumerate(batch[3]):
                        context_tag[i, :len(doc)] = torch.LongTensor(doc)

                    context_ent = torch.LongTensor(batch_size, context_len).fill_(0)
                    for i, doc in enumerate(batch[4]):
                        context_ent[i, :len(doc)] = torch.LongTensor(doc)
                else:
                    # create one-hot vectors
                    context_tag = torch.Tensor(batch_size, context_len, self.opt['pos_size']).fill_(0)
                    for i, doc in enumerate(batch[3]):
                        for j, tag in enumerate(doc):
                            context_tag[i, j, tag] = 1

                    context_ent = torch.Tensor(batch_size, context_len, self.opt['ner_size']).fill_(0)
                    for i, doc in enumerate(batch[4]):
                        for j, ent in enumerate(doc):
                            context_ent[i, j, ent] = 1

                question_len = max(len(x) for x in batch[5])
                question_id = torch.LongTensor(batch_size, question_len).fill_(0)
                for i, doc in enumerate(batch[5]):
                    question_id[i, :len(doc)] = torch.LongTensor(doc)

                context_mask = torch.eq(context_id, 0)
                question_mask = torch.eq(question_id, 0)
                text = list(batch[6])
                span = list(batch[7])
                context_sentence_lens = list(batch[8])

                if self.with_cids:
                    context_char_id = torch.LongTensor(batch_size, context_len, 50).fill_(260) # 260 is padding_char
                    for i,d in enumerate(batch[9]):
                        context_char_id[i, :d.size(0)] = d
                    question_char_id = torch.LongTensor(batch_size, question_len, 50).fill_(260)
                    for i,d in enumerate(batch[10]):
                        question_char_id[i, :d.size(0)] = d
                else:
                    context_char_id, question_char_id = None, None

                if not self.eval:
                    tmp = torch.LongTensor([ex[0] for ex in batch[-1]])
                    y_s = tmp[:, 0].contiguous()
                    y_e = tmp[:, 1].contiguous()
                elif context_char_id is not None:
                    context_char_id.volatile = True
                    question_char_id.volatile = True
                if self.gpu:
                    context_id = context_id.pin_memory()
                    context_feature = context_feature.pin_memory() if context_feature is not None else None
                    context_tag = context_tag.pin_memory()
                    context_ent = context_ent.pin_memory()
                    context_mask = context_mask.pin_memory()
                    question_id = question_id.pin_memory()
                    question_mask = question_mask.pin_memory()
                    context_char_id = context_char_id.cuda() if context_char_id is not None else None
                    question_char_id = question_char_id.cuda() if question_char_id is not None else None
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, context_sentence_lens, context_char_id, question_char_id, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, context_sentence_lens, context_char_id, question_char_id, y_s, y_e, text, span)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1

def get_inv_group(group_id):
    inv_group = {}
    for i, g in enumerate(group_id):
        if g in inv_group:
            inv_group[g].append(i)
        else:
            inv_group[g] = [i]
    return inv_group


