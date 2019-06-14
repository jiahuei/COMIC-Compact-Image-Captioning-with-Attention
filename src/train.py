#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:45:51 2017

@author: jiahuei

BUILT AGAINST TENSORFLOW r1.9.0

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, argparse
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURR_DIR, '..'))
sys.path.append(os.path.join(CURR_DIR, '..', 'common'))
import train_fn as train
import common.net_params as net_params
import common.utils as utils
pjoin = os.path.join


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument(
        '--name', type=str, default='lstm',
        help='The logging name.')
    parser.add_argument(
        '--dataset_dir', type=str, default='',
        help='The dataset directory.')
    parser.add_argument(
        '--dataset_file_pattern', type=str,
        default='mscoco_{}_w5_s20_include_restval',
        help='The dataset text files naming pattern.')
    parser.add_argument(
        '--train_mode', type=str, default='decoder',
        choices=['decoder', 'cnn_finetune', 'scst'],
        help='Str. The training regime.')
    
    parser.add_argument(
        '--legacy', type=bool, default=False,
        help='If True, will match settings as described in paper.')
    parser.add_argument(
        '--token_type', type=str, default='radix',
        choices=['radix', 'word', 'char'],
        help='The language model.')
    parser.add_argument(
        '--radix_base', type=int, default=256,
        help='The base for Radix models.')
    
    parser.add_argument(
        '--cnn_name', type=str, default='inception_v1',
        help='The CNN model name.')
    parser.add_argument(
        '--cnn_input_size', type=str, default='224,224',
        help='The network input size.')
    parser.add_argument(
        '--cnn_input_augment', type=bool, default=True,
        help='Whether to augment input images.')
    parser.add_argument(
        '--cnn_fm_attention', type=str, default='Mixed_4f',
        help='String, name of feature map for attention.')
    parser.add_argument(
        '--cnn_fm_projection', type=str, default='tied',
        choices=['none', 'independent', 'tied'],
        help='String, feature map projection, from `none`, `independent`, `tied`.')
    
    parser.add_argument(
        '--rnn_name', type=str, default='LSTM',
        choices=['LSTM', 'LN_LSTM', 'GRU'],
        help='The type of RNN, from `LSTM`, `LN_LSTM` and `GRU`.')
    parser.add_argument(
        '--rnn_size', type=int, default=512,
        help='Int, number of RNN units.')
    parser.add_argument(
        '--rnn_word_size', type=int, default=256,
        help='The word size.')
    parser.add_argument(
        '--rnn_init_method', type=str, default='first_input',
        choices=['project_hidden', 'first_input'],
        help='The RNN init method.')
    parser.add_argument(
        '--rnn_recurr_dropout', type=bool, default=False,
        help='Whether to enable variational recurrent dropout.')
    
    parser.add_argument(
        '--attn_num_heads', type=int, default=8,
        help='The number of attention heads.')
    parser.add_argument(
        '--attn_context_layer', type=bool, default=False,
        help='If True, add linear projection after multi-head attention.')
    parser.add_argument(
        '--attn_alignment_method', type=str, default='add_LN',
        choices=['add_LN', 'add', 'dot'],
        help='Str, The alignment method / composition method.')
    parser.add_argument(
        '--attn_probability_fn', type=str, default='softmax',
        choices=['softmax', 'sigmoid'],
        help='Str, The attention map probability function.')
    parser.add_argument(
        '--attn_keep_prob', type=float, default=0.9,
        help='Float, The keep rate for attention map dropout.')
    
    parser.add_argument(
        '--initialiser', type=str, default='xavier',
        choices=['xavier', 'he', 'none'],
        help='The initialiser: `xavier`, `he`, tensorflow default.')
    parser.add_argument(
        '--optimiser', type=str, default='adam',
        choices=['adam', 'sgd'],
        help='The optimiser: `adam`, `sgd`.')
    parser.add_argument(
        '--batch_size_train', type=int, default=32,
        help='The batch size for training.')
    # Divisors of 25010: 1, 2, 5, 10, 41, 61, 82, 122, 205, 305, 410, 610, 2501, 5002, 12505, 25010
    parser.add_argument(
        '--batch_size_eval', type=int, default=61,
        help='The batch size for validation.')
    parser.add_argument(
        '--max_epoch', type=int, default=30,
        help='The max epoch training.')
    parser.add_argument(
        '--lr_start', type=float, default=1e-2,
        help='Float, determines the starting learning rate.')
    parser.add_argument(
        '--lr_end', type=float, default=1e-5,
        help='Float, determines the ending learning rate.')
    parser.add_argument(
        '--cnn_grad_multiplier', type=float, default=1.0,
        help='Float, determines the gradient multiplier when back-prop thru CNN.')
    parser.add_argument(
        '--adam_epsilon', type=float, default=1e-2,
        help='Float, determines the epsilon value of ADAM.')
    parser.add_argument(
        '--scst_beam_size', type=int, default=7,
        help='The beam size for SCST sampling.')
    parser.add_argument(
        '--scst_weight_ciderD', type=float, default=1.0,
        help='The weight for CIDEr-D metric during SCST training.')
    parser.add_argument(
        '--scst_weight_bleu', type=str, default='0,0,0,2',
        help='The weight for BLEU metrics during SCST training.')
    
    parser.add_argument(
        '--freeze_scopes', type=str, default='Model/encoder/cnn',
        help='The scopes to freeze / do not train.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='The checkpoint path.')
    parser.add_argument(
        '--checkpoint_exclude_scopes', type=str, default='',
        help='The scopes to exclude when restoring from checkpoint.')
    parser.add_argument(
        '--gpu', type=str, default='0',
        help='The gpu number.')
    parser.add_argument(
        '--run', type=int, default=1,
        help='The run number.')
    
    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    for k in ['cnn_input_size']:
        args.__dict__[k] = [int(v) for v in args.__dict__[k].split(',')]
    
    if args.legacy:
        print('LEGACY mode enabled. Some arguments will be overridden.')
        args.cnn_name = 'inception_v1'
        args.cnn_input_size = '224,224'
        args.cnn_input_augment = True
        args.cnn_fm_attention = 'Mixed_4f'
        args.rnn_name = 'LSTM'
        args.rnn_size = 512
        args.rnn_word_size = 256
        args.rnn_init_method = 'project_hidden'
        args.rnn_recurr_dropout = False
        args.attn_context_layer = False
        args.attn_alignment_method = 'add_LN'
        args.attn_probability_fn = 'softmax'
        args.attn_keep_prob = 1.0
        args.lr_start = 1e-3
        args.lr_end = 2e-4
        args.lr_reduce_every_n_epochs = 4
        args.cnn_grad_multiplier = 1.0
        args.initialiser = 'xavier'
        args.optimiser = 'adam'
        args.batch_size_train = 32
        args.adam_epsilon = 1e-6
    
    if args.run == 1:
        rand_seed = 48964896
    elif args.run == 2:
        rand_seed = 88888888
    elif args.run == 3:
        rand_seed = 123456789
    
    dataset = args.dataset_file_pattern.split('_')[0]
    log_root = pjoin(os.path.dirname(CURR_DIR), 'experiments', dataset)
    if args.dataset_dir == '':
        args.dataset_dir = pjoin(os.path.dirname(CURR_DIR), 'datasets', dataset)
    
    if args.token_type == 'radix':
        token = 'radix_b{}'.format(args.radix_base)
    else:
        token = args.token_type
    name = '_'.join([
            token,
            args.attn_alignment_method,
            args.attn_probability_fn,
            'h{}'.format(args.attn_num_heads),
            args.cnn_fm_projection[:3],
            args.name,
            ])
    if args.legacy:
        name = 'legacy_' + name
    
    dec_dir = pjoin(log_root, '{}_run_{:02d}'.format(name, args.run))
    cnnft_dir = pjoin(log_root, '{}_cnnFT_run_{:02d}'.format(name, args.run))
    train_fn = train.train_fn
    
    if args.train_mode == 'decoder':
        assert args.freeze_scopes == 'Model/encoder/cnn'
        # Maybe download weights
        net = net_params.get_net_params(args.cnn_name, ckpt_dir_or_file=args.checkpoint_path)
        utils.maybe_get_ckpt_file(net)
        args.checkpoint_path = net['ckpt_path']
        log_path = dec_dir
    
    elif args.train_mode == 'cnn_finetune':
        # CNN fine-tune
        if args.legacy: raise NotImplementedError
        if not os.path.exists(dec_dir):
            raise ValueError('Decoder training log path not found: {}'.format(dec_dir))
        args.lr_start = 1e-3
        args.max_epoch = 10
        args.freeze_scopes = ''
        args.checkpoint_path = dec_dir
        log_path = cnnft_dir
    
    elif args.train_mode == 'scst':
        # SCST fine-tune (after CNN fine-tune)
        if args.legacy: raise NotImplementedError
        if not os.path.exists(cnnft_dir):
            raise ValueError('CNN finetune log path not found: {}'.format(cnnft_dir))
        args.scst_weight_bleu = [float(w) for w in args.scst_weight_bleu.split(',')]
        args.batch_size_train = 10
        args.lr_start = 1e-3
        args.max_epoch = 10
        args.freeze_scopes = 'Model/encoder/cnn'
        args.checkpoint_path = cnnft_dir
        scst = 'beam_{}_CrD_{}_B1_{}_B4_{}'.format(
                    args.scst_beam_size,
                    args.scst_weight_ciderD,
                    args.scst_weight_bleu[0], args.scst_weight_bleu[-1])
        scst_dir= pjoin(log_root, '{}_cnnFT_SCST_{}_run_{:02d}'.format(
                                            name, scst, args.run))
        log_path = scst_dir
        train_fn = train.train_fn_scst
    
    args.resume_training = overwrite = os.path.exists(log_path)
    
    ###
    
    # NoneType checking and conversion
    for k, v in args.__dict__.iteritems():
        if v == 'none':
            args.__dict__[k] = None
    
    kwargs = dict(
        rnn_layers = 1,
        dropout_rnn_in = 0.35,
        dropout_rnn_out = 0.35,
        rnn_map_loss_scale = 1.0,
        l2_decay = 1e-5,
        clip_gradient_norm = 0,
        
        max_saves = 12,
        num_logs_per_epoch = 100,
        per_process_gpu_memory_fraction = None,
        
        rand_seed = rand_seed,
        add_image_summaries = True,
        add_vars_summaries = False,
        add_grad_summaries = False,
        
        log_path = log_path,
        save_path = pjoin(log_path, 'model'),
        )
    
    kwargs.update(args.__dict__)
    
    ###
    
    train.try_to_train(
            train_fn = train_fn,
            try_block = True,
            overwrite = overwrite,
            **kwargs)


