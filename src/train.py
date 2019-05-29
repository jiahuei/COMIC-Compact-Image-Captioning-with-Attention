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

import train_fn as train
import os, sys, argparse
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURR_DIR, '..', 'common'))
import net_params
import utils
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
        '--attn_alignment_method', type=str, default='add',
        choices=['add', 'dot'],
        help='Str, The alignment method / composition method.')
    parser.add_argument(
        '--attn_probability_fn', type=str, default='softmax',
        choices=['softmax', 'sigmoid'],
        help='Str, The attention map probability function.')
    
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
        '--adam_epsilon', type=float, default=1e-2,
        help='Float, determines the epsilon value of ADAM.')
    
    parser.add_argument(
        '--freeze_scopes', type=str, default='Model/encoder/cnn',
        help='The scopes to freeze / do not train.')
    parser.add_argument(
        '--resume_training', type=bool, default=False,
        help='Boolean, whether to resume training from checkpoint. Pass '' for False.')
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
    #overwrite = args.resume_training
    overwrite = True
    
    for k in ['cnn_input_size']:
        args.__dict__[k] = [int(v) for v in args.__dict__[k].split(',')]
    
    if args.run == 1:
        rand_seed = 48964896
    elif args.run == 2:
        rand_seed = 88888888
    elif args.run == 3:
        rand_seed = 123456789
    
    dataset = args.dataset_file_pattern.split('_')[0]
    log_root = pjoin(os.path.dirname(CURR_DIR), 'experiments', dataset)
    if args.dataset_dir == '':
        args.dataset_dir = pjoin(CURR_DIR, 'datasets', dataset)
    
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
    log_path = pjoin(log_root, '{}_run_{:02d}'.format(name, args.run))
    cnn_ft_log = '{}_cnnFT_run_{:02d}'.format(name, args.run)
    cnn_ft_log = pjoin(log_root, cnn_ft_log)
    train_fn = train.train_fn
    
    if not os.path.exists(log_path):
        # Maybe download weights
        net = net_params.get_net_params(args.cnn_name)
        utils.maybe_get_ckpt_file(net)
        args.checkpoint_path = net['ckpt_path']
    elif os.path.exists(log_path) and not os.path.exists(cnn_ft_log):
        # CNN fine-tune
        args.lr_start = 1e-3
        args.max_epoch = 10
        args.freeze_scopes = ''
        args.checkpoint_path = log_path
        log_path = pjoin(log_root, cnn_ft_log)
    
    elif os.path.exists(cnn_ft_log):
        # SCST fine-tune (after CNN fine-tune)
        raise ValueError('Not ready')
        log_path = pjoin(log_root, log_name)
        train_fn = train.train_fn_scst
    
    
    ###
    
    # NoneType checking and conversion
    for k, v in args.__dict__.iteritems():
        if v == 'none':
            args.__dict__[k] = None
    
    kwargs = dict(
        rnn_layers = 1,
        #infer_beam_size = 3,        # not used
        #infer_max_length = 40,      # not used
        
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
            train_fn = train.train_fn,
            try_block = True,
            overwrite = overwrite,
            **kwargs)




'''

###############################################################################
#                             VGG-16 end points                               #
###############################################################################

'vgg_16/conv1/conv1_1'
'vgg_16/conv1/conv1_2'
'vgg_16/conv2/conv2_1'
'vgg_16/conv2/conv2_2'
'vgg_16/conv3/conv3_1'
'vgg_16/conv3/conv3_2'
'vgg_16/conv3/conv3_3'
'vgg_16/conv4/conv4_1'
'vgg_16/conv4/conv4_2'
'vgg_16/conv4/conv4_3'
'vgg_16/conv5/conv5_1'
'vgg_16/conv5/conv5_2'
'vgg_16/conv5/conv5_3'
'vgg_16/pool1'
'vgg_16/pool2'
'vgg_16/pool3'
'vgg_16/pool4'
'vgg_16/pool5'
'vgg_16/fc6'
'vgg_16/fc7'


###############################################################################
#                        Inception-v1 end points                              #
###############################################################################

'Conv2d_1a_7x7'
'MaxPool_2a_3x3'
'Conv2d_2b_1x1'
'Conv2d_2c_3x3'
'MaxPool_3a_3x3'
'Mixed_3b'
'Mixed_3c'                  # 28 x 28 x 480
'MaxPool_4a_3x3'            # 14 x 14 x 832
'Mixed_4b'
'Mixed_4c'
'Mixed_4d'
'Mixed_4e'
'Mixed_4f'                  # 14 x 14 x 832
'MaxPool_5a_2x2'            # 7 x 7 x 1024
'Mixed_5b'
'Mixed_5c'                  # 7 x 7 x 1024


###############################################################################
#                        Inception-v3 end points                              #
###############################################################################

'Conv2d_1a_3x3'
'Conv2d_2a_3x3'
'Conv2d_2b_3x3'
'MaxPool_3a_3x3'
'Conv2d_3b_1x1'
'Conv2d_4a_3x3'
'MaxPool_5a_3x3'
'Mixed_5b'
'Mixed_5c'
'Mixed_5d'
'Mixed_6a'
'Mixed_6b'
'Mixed_6c'
'Mixed_6d'
'Mixed_6e'
'Mixed_7a'
'Mixed_7b'
'Mixed_7c'



'''




