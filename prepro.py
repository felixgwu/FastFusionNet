# Origin: https://github.com/taolei87/sru/blob/master/DrQA/prepro.py 
# Modified by Felix Wu

import torch
import re
import json
import spacy
# import msgpack
import unicodedata
import numpy as np
import pandas as pd
import argparse
import collections
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
from os.path import join
from qa.utils import str2bool
from qa.general_utils import normalize_text, build_embedding, load_glove_vocab, feature_gen, token2id, process_jsonlines


parser = argparse.ArgumentParser(
    description='Preprocessing data files, about 10 minitues to run.'
)
parser.add_argument('--wv_file', default='data/glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--wv_cased', type=str2bool, default=True,
                    help='treat the words as cased or not.')
parser.add_argument('--sort_all', type=str2bool, default=True,
                    help='sort the vocabulary by frequencies of all words. '
                         'Otherwise consider question words first.')
parser.add_argument('--sample_size', type=int, default=100,
                    help='size of sample data (for debugging).')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size for multiprocess tokenizing and tagging.')
parser.add_argument('--data_dir', type=str, default='data/squad',
                    help='data directory.')
parser.add_argument('--pre_proc', default='fusion', help='[sru/fusion/qanet]')
parser.add_argument('--train', default='data/squad/train-v1.1.json')
parser.add_argument('--dev', default='data/squad/dev-v1.1.json')
parser.add_argument('--suffix', type=str, default='fusion',
                    help='suffix of the output file')

args = parser.parse_args()
trn_file = args.train
dev_file = args.dev
wv_file = args.wv_file
wv_dim = args.wv_dim

if args.pre_proc == 'sru':
    from qa.general_utils import pre_proc_sru as pre_proc
elif args.pre_proc == 'fusion':
    from qa.general_utils import pre_proc_fusion as pre_proc
elif args.pre_proc == 'qanet':
    from qa.general_utils import pre_proc_qanet as pre_proc
else:
    raise ValueError('args.pre_proc={}'.format(args.pre_proc))

args.spacy_version = spacy.__version__

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info(vars(args))
log.info('start data preparing...')


wv_vocab = load_glove_vocab(wv_file, wv_dim)
log.info('glove loaded.')


def flatten_json(file, proc_func):
    '''A multi-processing wrapper for loading SQuAD data file.'''
    with open(file) as f:
        data = json.load(f)['data']
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        rows = executor.map(proc_func, data)
    rows = sum(rows, [])
    return rows


def proc_train(article):
    '''Flatten each article in training data.'''
    rows = []
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            answer_starts = [a['answer_start'] for a in answers]
            answers = [a['text'] for a in answers]
            answer_ends = [answer_start + len(answer) for answer_start, answer in zip(answer_starts, answers)]
            rows.append((id_, context, question, answers, answer_starts, answer_ends))
    return rows


def proc_dev(article):
    '''Flatten each article in dev data'''
    rows = []
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            answers = [a['text'] for a in answers]
            rows.append((id_, context, question, answers))
    return rows


train = flatten_json(trn_file, proc_train)
train = pd.DataFrame(train,
                     columns=['id', 'context', 'question', 'answers',
                              'answer_starts', 'answer_ends'])
dev = flatten_json(dev_file, proc_dev)
dev = pd.DataFrame(dev,
                   columns=['id', 'context', 'question', 'answers'])
##debug
# train = train[:100]
# dev = dev[:100]

log.info('json data flattened.')

nlp = spacy.load('en', parser=False, tagger=False, entity=False)

# indices = [28, 40, 86]
# for i in indices:
#     print(train.context[i])

context_iter = (pre_proc(c) for c in train.context)
context_tokens = [[w.text for w in doc] for doc in nlp.pipe(
    context_iter, batch_size=args.batch_size, n_threads=args.threads)]
log.info('got intial tokens.')


def get_answer_index(context, context_token, answer_starts, answer_ends):
    '''
    Get exact indices of the answer in the tokens of the passage,
    according to the start and end position of the answer.

    Args:
        context (str): the context passage
        context_token (list): list of tokens (str) in the context passage
        answer_starts (list): the start position of the answer in the passage
        answer_ends (list): the end position of the answer in the passage

    Returns:
        (int, int): start index and end index of answer
    '''
    p_str = 0
    p_token = 0
    start_tokens, end_tokens = [], []
    for answer_start, answer_end in zip(answer_starts, answer_ends):
        while p_str < len(context):
            if re.match('\s', context[p_str]):
                p_str += 1
                continue
            token = context_token[p_token]
            token_len = len(token)
            if context[p_str:p_str + token_len] != token:
                return (None, None)
            if p_str == answer_start:
                t_start = p_token
            p_str += token_len
            if p_str == answer_end:
                try:
                    start_tokens.append(t_start)
                    end_tokens.append(p_token)
                except UnboundLocalError as e:
                    pass
                finally:
                    break
                
            p_token += 1
    if len(start_tokens) == 0:
        return (None, None)
    else:
        return (start_tokens, end_tokens)

train['answer_start_tokens'], train['answer_end_tokens'] = \
    zip(*[get_answer_index(a, b, c, d) for a, b, c, d in
          zip(train.context, context_tokens,
              train.answer_starts, train.answer_ends)])
initial_len = len(train)
train.dropna(inplace=True)
log.info('drop {} inconsistent samples.'.format(initial_len - len(train)))
log.info('answer pointer generated.')

ntrain = len(train)
ndev = len(dev)
questions = list(train.question) + list(dev.question)
contexts = list(train.context) + list(dev.context)

nlp = spacy.load('en', parser=False)
context_text = [pre_proc(c) for c in contexts]
question_text = [pre_proc(q) for q in questions]
question_docs = [doc for doc in nlp.pipe(
    iter(question_text), batch_size=args.batch_size, n_threads=args.threads)]
context_docs = [doc for doc in nlp.pipe(
    iter(context_text), batch_size=args.batch_size, n_threads=args.threads)]
log.info('parsed')


# get tokens
if args.wv_cased:
    question_tokens = [[normalize_text(w.text) for w in doc] for doc in question_docs]
    context_tokens = [[normalize_text(w.text) for w in doc] for doc in context_docs]
else:
    question_tokens = [[normalize_text(w.text).lower() for w in doc] for doc in question_docs]
    context_tokens = [[normalize_text(w.text).lower() for w in doc] for doc in context_docs]

def get_spans(tokens, text):
    pos = 0
    spans = []
    for token in tokens:
        start = pos + text[pos:].find(token)
        spans.append([start, start+len(token)])
        pos = start + len(token)
    for (s, e), token in zip(spans, tokens):
        assert text[s:e] == token, '{}, {}\ntext: {}\n token: {}'.foramt(s, e, text, token)
    return spans

context_token_span = [get_spans([w.text for w in doc], text) for doc, text in zip(context_docs, contexts)]
question_token_span = [get_spans([w.text for w in doc], text) for doc, text in zip(question_docs, questions)]
# context_token_span = [[(w.idx, w.idx + len(w.text)) for w in doc] for doc in context_docs]
# context_sentence_lens = [[len(sent) for sent in doc.sents] for doc in context_docs]
context_sentence_lens = [[] for doc in context_docs]
log.info('tokens generated')


# get features
context_tags = [[w.tag_ for w in doc] for doc in context_docs]
context_ents = [[w.ent_type_ for w in doc] for doc in context_docs]
context_features = []
for question, context in zip(question_docs, context_docs):
    question_word = {w.text for w in question}
    question_lower = {w.text.lower() for w in question}
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
    match_origin = [w.text in question_word for w in context]
    match_lower = [w.text.lower() in question_lower for w in context]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context]
    context_features.append(list(zip(match_origin, match_lower, match_lemma)))
log.info('features generated')


def build_train_vocab(questions, contexts, wv_vocab): # vocabulary will also be sorted accordingly
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get, reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                        key=counter.get, reverse=True)

    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab {1}/{0} OOV {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab

def build_eval_vocab(questions, contexts, train_vocab, wv_vocab): # most vocabulary comes from tr_vocab
    existing_vocab = set(train_vocab)
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab and w in wv_vocab]))
    vocab = train_vocab + new_vocab
    log.info('train vocab {0}, total vocab {1}'.format(len(train_vocab), len(vocab)))
    return vocab

def build_full_vocab(questions, contexts, eval_vocab):
    existing_vocab = set(eval_vocab)
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab]))
    vocab = eval_vocab + new_vocab
    log.info('eval vocab {0}, total vocab {1}'.format(len(eval_vocab), len(vocab)))
    return vocab

# vocab
train_vocab = build_train_vocab(question_tokens[:ntrain], context_tokens[:ntrain], wv_vocab)
eval_vocab = build_eval_vocab(question_tokens[ntrain:], context_tokens[ntrain:], train_vocab, wv_vocab)
# train_context_ids = token2id(context_tokens[:ntrain], train_vocab, unk_id=1)
# train_question_ids = token2id(question_tokens[:ntrain], train_vocab, unk_id=1)
# full_vocab = set(w for doc in context_tokens + question_tokens for w in doc) | set(eval_vocab)

# tokens
question_ids = token2id(question_tokens, eval_vocab, unk_id=1)
context_ids = token2id(context_tokens, eval_vocab, unk_id=1)


# term frequency in document
context_tf = []
for doc in context_tokens:
    counter_ = collections.Counter(w.lower() for w in doc)
    total = sum(counter_.values())
    context_tf.append([counter_[w.lower()] / total for w in doc])
context_features = [[list(w) + [tf] for w, tf in zip(doc, tfs)] for doc, tfs in
                    zip(context_features, context_tf)]

# tags
vocab_tag = list(nlp.tagger.tag_names)
context_tag_ids = token2id(context_tags, vocab_tag)
log.info('Found {} POS tags.'.format(len(vocab_tag)))

# entities, build dict on the fly
vocab_ent = [''] + nlp.entity.cfg[u'actions']['1']
context_ent_ids = token2id(context_ents, vocab_ent)
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))

log.info('vocab built.')


embedding = build_embedding(wv_file, eval_vocab, wv_dim)
log.info('got embedding matrix.')




# train.to_csv('data/squad/train.csv', index=False)
# dev.to_csv('data/squad/dev.csv', index=False)

train_feature_names = ['id', 'context_word_ids', 'context_features', 'context_pos_ids', 'context_ent_ids',
                       'question_word_ids',
                       'context', 'context_token_span', 'context_sentence_lens',
                       'question', 'question_token_span',
                       'answers', 'answer_spans']
dev_feature_names = ['id', 'context_word_ids', 'context_features', 'context_pos_ids', 'context_ent_ids',
                     'question_word_ids',
                     'context', 'context_token_span', 'context_sentence_lens',
                     'question', 'question_token_span',
                     'answers']

meta = {
    'prepro_args': dict(vars(args)),
    'vocab': eval_vocab,
    'train_vocab_size': len(train_vocab),
    'vocab_tag': vocab_tag,
    'vocab_ent': vocab_ent,
    'embedding': embedding.tolist(),
    'train_feature_names': train_feature_names,
    'dev_feature_names': dev_feature_names,
    'args': dict(vars(args)),
    # 'full_vocab': full_vocab,
}

train_new = list(zip(
    train['id'].tolist(),
    context_ids[:ntrain],
    context_features[:ntrain],
    context_tag_ids[:ntrain],
    context_ent_ids[:ntrain],
    question_ids[:ntrain],
    contexts[:ntrain], # context_text[:ntrain],
    context_token_span[:ntrain],
    context_sentence_lens[:ntrain],
    questions[:ntrain], # question_text[:ntrain],
    question_token_span[:ntrain],
    train.answers.tolist(),
    [list(zip(s, e)) for s, e in zip(train['answer_start_tokens'], train['answer_end_tokens'])],
))
dev_new = list(zip(
    dev['id'].tolist(),
    context_ids[ntrain:],
    context_features[ntrain:],
    context_tag_ids[ntrain:],
    context_ent_ids[ntrain:],
    question_ids[ntrain:],
    contexts[ntrain:], # context_text[ntrain:],
    context_token_span[ntrain:],
    context_sentence_lens[ntrain:],
    questions[ntrain:], # question_text[ntrain:],
    question_token_span[ntrain:],
    dev['answers'].tolist(),
))

for i, d in enumerate(train_new):
    if not d[6][d[7][d[12][0][0]][0]:d[7][d[12][0][1]][1]] == d[11][0]:
        print(i, d[6][d[7][d[12][0][0]][0]:d[7][d[12][0][1]][1]], d[11][0])
print(len(train_new), len(dev_new))

result = {
    'train': train_new,
    'dev': dev_new
}

torch.save({'data': result, 'meta': meta}, join(args.data_dir, 'data{}.pth'.format('' if args.suffix == '' else '-' + args.suffix)))

if args.sample_size:
    sample = {
        'train': train_new[:args.sample_size],
        'dev': dev_new[:args.sample_size]
    }
    # with open(join(args.data_dir, 'sample-{}.msgpack'.format(args.sample_size)), 'wb') as f:
        # msgpack.dump(sample, f)
    torch.save({'data': sample, 'meta': meta}, join(args.data_dir, 'sample-{}{}.pth'.format(args.sample_size, '' if args.suffix == '' else '-' + args.suffix)))
log.info('saved to disk.')
# with open('data/squad/meta.msgpack', 'wb') as f:
#     msgpack.dump(meta, f)
# result = {
#     'trn_question_ids': question_ids[:len(train)],
#     'dev_question_ids': question_ids[len(train):],
#     'trn_context_ids': context_ids[:len(train)],
#     'dev_context_ids': context_ids[len(train):],
#     'trn_context_features': context_features[:len(train)],
#     'dev_context_features': context_features[len(train):],
#     'trn_context_tags': context_tag_ids[:len(train)],
#     'dev_context_tags': context_tag_ids[len(train):],
#     'trn_context_ents': context_ent_ids[:len(train)],
#     'dev_context_ents': context_ent_ids[len(train):],
#     'trn_context_text': context_text[:len(train)],
#     'dev_context_text': context_text[len(train):],
#     'trn_context_spans': context_token_span[:len(train)],
#     'dev_context_spans': context_token_span[len(train):]
# }
# with open('data/squad/data.msgpack', 'wb') as f:
#     msgpack.dump(result, f)
# if args.sample_size:
#     sample_size = args.sample_size
#     sample = {
#         'trn_question_ids': result['trn_question_ids'][:sample_size],
#         'dev_question_ids': result['dev_question_ids'][:sample_size],
#         'trn_context_ids': result['trn_context_ids'][:sample_size],
#         'dev_context_ids': result['dev_context_ids'][:sample_size],
#         'trn_context_features': result['trn_context_features'][:sample_size],
#         'dev_context_features': result['dev_context_features'][:sample_size],
#         'trn_context_tags': result['trn_context_tags'][:sample_size],
#         'dev_context_tags': result['dev_context_tags'][:sample_size],
#         'trn_context_ents': result['trn_context_ents'][:sample_size],
#         'dev_context_ents': result['dev_context_ents'][:sample_size],
#         'trn_context_text': result['trn_context_text'][:sample_size],
#         'dev_context_text': result['dev_context_text'][:sample_size],
#         'trn_context_spans': result['trn_context_spans'][:sample_size],
#         'dev_context_spans': result['dev_context_spans'][:sample_size]
#     }
#     with open('data/squad/sample.msgpack', 'wb') as f:
#         msgpack.dump(sample, f)
# log.info('saved to disk.')

