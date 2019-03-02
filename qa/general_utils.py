# Modified from https://github.com/momohuang/FusionNet-NLI/blob/master/general_utils.py

import re
import os
import sys
import random
import string
import logging
import argparse
import unicodedata
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import jsonlines
import numpy as np

#===========================================================================
#================= All for preprocessing SQuAD data set ====================
#===========================================================================

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def load_glove_vocab(file, wv_dim):
    vocab = set()
    with open(file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            vocab.add(token)
    return vocab

def pre_proc_sru(text):
    '''normalize spaces in a string. From SRU DrQA code'''
    text = re.sub('\s+', ' ', text)
    return text

def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '

def pre_proc_fusion(text):
    '''from FusionNet-NLI'''
    # make hyphens, spaces clean
    text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

extra_split_chars = (u'-', u'£', u'€', u'¥', u'¢', u'₹', u'\u2212', u'\u2014',
                     u'\u2013', u'/', u'~', u'"', u"'", u'\ud01C', u'\u2019',
                     u'\u201D', u'\u2018', u'\u00B0')
extra_split_tokens = (
    u'``',
    u'(?<=[^_])_(?=[^_])',  # dashes w/o a preceeding or following dash, so __wow___ -> ___ wow ___
    u"''",
    u'[' + u''.join(extra_split_chars) + ']')
extra_split_chars_re = re.compile(u'(' + u'|'.join(extra_split_tokens) + u')')

def pre_proc_qanet(text):
    '''from QANet code'''
    # make hyphens, spaces clean
    text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
    text = extra_split_chars_re.sub(space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

def process_jsonlines(data_file):
    with jsonlines.open(data_file) as reader:
        snli_label = []
        snli_sent1 = []
        snli_sent2 = []
        for obj in reader:
            if obj['gold_label'] != '-':
                snli_label.append(obj['gold_label'])
                snli_sent1.append(obj['sentence1'])
                snli_sent2.append(obj['sentence2'])
        return SNLIData(snli_label, snli_sent1, snli_sent2)

def feature_gen(A_docs, B_docs):
    A_tags = [[w.tag_ for w in doc] for doc in A_docs]
    A_ents = [[w.ent_type_ for w in doc] for doc in A_docs]
    A_features = []

    for textA, textB in zip(A_docs, B_docs):
        counter_ = Counter(w.text.lower() for w in textA)
        total = sum(counter_.values())
        term_freq = [counter_[w.text.lower()] / total for w in textA]

        question_word = {w.text for w in textB}
        question_lower = {w.text.lower() for w in textB}
        question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in textB}
        match_origin = [w.text in question_word for w in textA]
        match_lower = [w.text.lower() in question_lower for w in textA]
        match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in textA]
        A_features.append(list(zip(term_freq, match_origin, match_lower, match_lemma)))

    return A_tags, A_ents, A_features

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

def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids

