import re
import os
import sys
import time
import json
import random
import logging
import argparse
import torch

from shutil import copyfile
from datetime import datetime
from collections import Counter
from tensorboardX import SummaryWriter

from qa.model import DocReaderModel
from qa.utils import *

parser = argparse.ArgumentParser(
    description='Train a QA model.'
)
# system
parser.add_argument('--log_file', default='output.log',
                    help='path for log file.')
parser.add_argument('--log_per_updates', type=int, default=50,
                    help='log model loss per x updates (mini-batches).')
parser.add_argument('--data_suffix', default='fusion',
                    help='suffix of the preprocessed data file.')
parser.add_argument('--save_dir', default='save/debug',
                    help='path to store saved models.')
parser.add_argument('--save_last_only', action='store_true',
                    help='only save the final models.')
parser.add_argument('--MTLSTM_path', default='data/glove/MT-LSTM.pth',
                    help='path to pretrained CoVe.')
parser.add_argument('--eval_per_epoch', type=int, default=1,
                    help='perform evaluation per x epochs.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument("--debug", action='store_true',
                    help='debug mode')
parser.add_argument('--profile',type = str,default = '', help = 'profile file name')
parser.add_argument('--profile_std',action = 'store_true')

# training
parser.add_argument('-e', '--epochs', type=int, default=40)
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-rs', '--resume', default='',
                    help='previous model file name (in `save_dir`). '
                         'e.g. "checkpoint_epoch_11.pt"')
parser.add_argument('-ro', '--resume_options', action='store_true',
                    help='use previous model options, ignore the cli and defaults.')
parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                    help='reduce initial (resumed) learning rate by this factor.')
parser.add_argument('--decay_every', type=int, default=0,
                    help='reduce learning rate very this many epochs.')
parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                    help='learning rate decay rate')
parser.add_argument('-op', '--optimizer', default='adamax',
                    help='supported optimizer: adamax, sgd, adam')
parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.002,
                    help='only applied to SGD.')
parser.add_argument('-mm', '--momentum', type=float, default=0,
                    help='only applied to SGD.')
parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                    help='finetune top-x embeddings.')
parser.add_argument('--fix_embeddings', action='store_true',
                    help='if true, `tune_partial` will be ignored.')
parser.add_argument('--rnn_padding', action='store_true',
                    help='perform rnn padding (much slower but more accurate).')
parser.add_argument('--max_train_len', type=int, default=0, help='max len for training')
parser.add_argument('--max_eval_len', type=int, default=0, help='max len for evaluation')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
parser.add_argument('--warm_start', action='store_true')
parser.add_argument('--ema_decay', type=float, default=1.,
                    help='Exponential moving average decay rate for trainable variables')
parser.add_argument('--valid_size', type=int, default=0)
parser.add_argument('--train_eval_size', type=int, default=3200, help='first n training data are used for evaluation')
parser.add_argument('--save_checkpoints', action='store_true')

# model
parser.add_argument('-m', '--model_type', default='drqa')
parser.add_argument('--question_merge', default='self_attn')
parser.add_argument('--doc_layers', type=int, default=3)
parser.add_argument('--question_layers', type=int, default=3)
parser.add_argument('--fusion_reading_layers', type=int, default=2)
parser.add_argument('--fusion_understanding_layers', type=int, default=1)
parser.add_argument('--fusion_final_layers', type=int, default=1)
parser.add_argument('--fusion_self_boost_times', type=int, default=1)
parser.add_argument('--fusion_gldr_layers', type=int, default=3)
parser.add_argument('--fusion_gldr_dilation_base', type=int, default=2)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_features', type=int, default=4)
parser.add_argument('--match', type=str2bool, nargs='?', const=True, default=True,
                    help='match features.')
parser.add_argument('--tf', type=str2bool, nargs='?', const=True, default=True,
                    help='term frequency features.')
parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=True,
                    help='use pos tags as a feature.')
parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=True,
                    help='use named entity tags as a feature.')
parser.add_argument('--use_word_emb', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--use_qemb', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--use_demb', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?',
                    const=True, default=True)
parser.add_argument('--word_dropout_c', type=float, default=0.0)
parser.add_argument('--word_dropout_q', type=float, default=0.0)
parser.add_argument('--dropout_emb', type=float, default=0.4)
parser.add_argument('--dropout_rnn', type=float, default=0.4)
parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?',
                    const=True, default=True)
parser.add_argument('--variational_dropout', type=str2bool, nargs='?',
                    const=True, default=True)
parser.add_argument('--depth_drop', type=float, default=0., help='max dropout rate of stochastic depth')
parser.add_argument('--max_len', type=int, default=15)
parser.add_argument('--rnn_type', default='lstm',
                    help='supported types: rnn, gru, lstm')
parser.add_argument('--pos_dim', type=int, default=12)
parser.add_argument('--ner_dim', type=int, default=8)
parser.add_argument('--use_feat_emb', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--end_gru', action='store_true')
parser.add_argument('--use_cove', action='store_true')
parser.add_argument('--use_max_char_emb', action='store_true')
parser.add_argument('--max_char_emb_size', type=int, default=200)
parser.add_argument('--max_char_emb_max_len', type=int, default=16)
parser.add_argument('--residual', action='store_true')
parser.add_argument('--squeeze_excitation', type=int, default=0, help='squeeze excitation reduction ratio')
parser.add_argument('--sentence_level', action='store_true')
args = parser.parse_args()

if not args.match:
    args.num_features -= 3
if not args.tf:
    args.num_features -= 1
if args.fix_embeddings:
    args.tune_partial = 0

# set model dir
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)
save_dir = os.path.abspath(save_dir)

# set random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(save_dir, args.log_file))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

writer = SummaryWriter(save_dir)

def main():
    log.info('[program starts.]')
    train, dev, train_y, dev_y, embedding, opt, meta = load_data(vars(args), log)
    # hold out original dev set
    log.info('[Data loaded.]')
    log.info('train_size: {}, dev_size: {}'.format(len(train), len(dev)))

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(save_dir, args.resume))
        if args.resume_options:
            opt.update(checkpoint['config'])
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        for i in range(checkpoint['epoch']):
            random.shuffle(list(range(len(train))))  # synchronize random seed
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1
    log.info('opt: {}'.format(opt))

    if args.cuda:
        model.cuda()

    log.info('model:\n{}'.format(model.network))

    opt['with_cids'] = False

    if args.profile:
        from pycallgraph import PyCallGraph, Config
        from pycallgraph.output import GraphvizOutput
        log.info('starts profiling')
        with PyCallGraph(output=GraphvizOutput(output_file=args.profile), config=Config(include_stdlib=args.profile_std)):
            batches = BatchGen(opt, dev, batch_size=args.batch_size, max_len=args.max_eval_len, evaluation=True, gpu=args.cuda, with_cids=opt['with_cids'])
            predictions = []
            for batch in batches:
                predictions.extend(model.predict(batch))
        print(len(dev_y), len(predictions))
        dev_em, dev_f1 = score(predictions, dev_y)
        log.info("[dev EM: {} F1: {}]".format(dev_em, dev_f1))
        log.info('finished profiling')
        return

    if args.resume:
        if not 'best_val_score' in checkpoint:
            batches = BatchGen(opt, dev, batch_size=args.batch_size, max_len=args.max_eval_len, evaluation=True, gpu=args.cuda, with_cids=opt['with_cids'])
            predictions = []
            for batch in batches:
                predictions.extend(model.predict(batch))
            em, f1 = score(predictions, dev_y)
            log.info("[dev EM: {} F1: {}]".format(em, f1))
            best_val_score = f1
        else:
            best_val_score = checkpoint['best_val_score']

    else:
        best_val_score = 0.

    log.info('best score is set to {:.2f}'.format(best_val_score))


    with open(os.path.join(save_dir, 'opt.json'), 'w') as f:
        json.dump(opt, f, indent=2)
    with open(os.path.join(save_dir, 'model_str.txt'), 'w') as f:
        print('model:\n{}\n\noptimizer:{}'.format(model.network, model.optimizer), file=f)


    dawn_log = os.path.join(args.save_dir, 'dawn_train.tsv')
    with open(dawn_log, 'w') as f:
        print('epoch\thours\tf1Score', file=f)
    all_train_time = 0.

    for epoch in range(epoch_0, args.epochs):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(opt, train, batch_size=args.batch_size, max_len=args.max_train_len, gpu=args.cuda, with_cids=opt['with_cids'])
        start = datetime.now()
        num_train_batches = len(batches)
        for i, batch in enumerate(batches):
            model.update(batch)
            if model.updates % args.log_per_updates == 0:
                log.info('epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    epoch, model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
                writer.add_scalar('train_loss_avg_iter', model.train_loss.avg, model.updates)
                writer.add_scalar('train_loss_iter', model.train_loss.val, model.updates)

        train_time = datetime.now() - start
        all_train_time += train_time.total_seconds()
        # eval
        if epoch % args.eval_per_epoch == 0:
            train_batches = BatchGen(opt, train[:args.train_eval_size], batch_size=args.batch_size, evaluation=True, max_len=args.max_eval_len, gpu=args.cuda, with_cids=opt['with_cids'])
            predictions = []
            for batch in train_batches:
                predictions.extend(model.predict(batch))
            train_em, train_f1 = score(predictions, train_y[:args.train_eval_size])

            dev_batches = BatchGen(opt, dev, batch_size=args.batch_size, evaluation=True, max_len=args.max_eval_len, gpu=args.cuda, with_cids=opt['with_cids'])
            predictions = []
            start = datetime.now()
            for batch in dev_batches:
                predictions.extend(model.predict(batch))
            dev_em, dev_f1 = score(predictions, dev_y)
            eval_time = datetime.now() - start

            is_best = best_val_score < dev_f1
            if is_best:
                best_val_score = dev_f1

            log.warning("Epoch {} train loss: {:.5f} EM: {:.2f} F1: {:.2f} dev EM: {:.2f} F1: {:.2f} (best: {:.2f}) train: {:.2f} s eval: {:.2f} s".format(epoch, model.train_loss.avg, train_em, train_f1, dev_em, dev_f1, best_val_score, train_time.total_seconds(), eval_time.total_seconds()))
            with open(dawn_log, 'a') as f:
                print('{}\t{}\t{}'.format(epoch, all_train_time / 3600., dev_f1), file=f)
            writer.add_scalar('train_loss_avg_epoch', model.train_loss.avg, epoch)
            writer.add_scalar('time_train', train_time.total_seconds(), epoch),
            writer.add_scalar('time_eval', eval_time.total_seconds(), epoch),
            writer.add_scalar('time_per_epoch_train', train_time.total_seconds() / num_train_batches, epoch),
            writer.add_scalar('time_per_epoch_eval', eval_time.total_seconds() / len(dev_batches), epoch)
            writer.add_scalar('EM_train', train_em, epoch)
            writer.add_scalar('F1_train', train_f1, epoch)
            writer.add_scalar('EM_dev', dev_em, epoch)
            writer.add_scalar('F1_dev', dev_f1, epoch)
            # save
            if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
                prev_model_file = os.path.join(save_dir, 'checkpoint_epoch_{}.pt'.format(epoch-1))
                model_file = os.path.join(save_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
                model.save(model_file, epoch, best_val_score)
                if is_best:
                    copyfile(
                        model_file,
                        os.path.join(save_dir, 'best_model.pt'))
                    log.info('[new best model saved.]')
                if os.path.exists(prev_model_file) and not args.save_checkpoints:
                    os.remove(prev_model_file)
        if args.decay_every > 0 and epoch % args.decay_every == 0:
            lr_decay(model.optimizer, args.lr_decay_rate)


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    log.info('[learning rate reduced by {}]'.format(lr_decay))
    return optimizer


if __name__ == '__main__':
    main()
