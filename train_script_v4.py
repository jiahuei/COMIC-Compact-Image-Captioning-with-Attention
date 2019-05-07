#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:45:51 2017

@author: jiahuei

BUILT AGAINST TENSORFLOW r1.2.1

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import train_fn_v4 as train
import os, argparse
pjoin = os.path.join

###############################################################################
# Change:
# - root
# - dropbox_log_path
# - log_name
# - lang_model
# - initialiser
# - rand_seed
# - per_process_gpu_memory_fraction
###############################################################################

root = '/ext_hdd'
root = '/home/jiahuei/Documents'
#root = '/home/chun/jh'
cnn_root = pjoin(root, '4_Pre_trained', 'tf_slim')

dropbox_log_path = '/home/jiahuei/Dropbox/@_PhD/Codes/TensorFlow_scripts/caption_baseN/dropbox_log'
#dropbox_log_path = '/home/jiahuei/Dropbox/@_PhD/Codes/TF_scripts_PowerEdge/caption_baseN/dropbox_log'
note = "meh.\r\n"

###


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="lol")
    
    parser.add_argument(
        '--name', type=str, required=True,
        help="The logging name.")
    parser.add_argument(
        '--lang_model', type=str, required=True,
        help="The language model, from 'baseN', 'word, 'char', 'bpe'.")
    parser.add_argument(
        '--image_model', type=str, required=True,
        help="The language model, from 'vgg_16', 'InceptionV1', 'InceptionV3', 'InceptionV4'.")
    parser.add_argument(
        '--base', type=int, default=128,
        help="The base for Base-N models.")
    parser.add_argument(
        '--fm_projection', type=str, required=True,
        help="String, feature map projection, from 'none', 'untied', 'tied'.")
    parser.add_argument(
        '--num_heads', type=int, default=None,
        help="Int, number of heads.")
    parser.add_argument(
        '--word_size', type=int, required=True,
        help="The word size.")
    parser.add_argument(
        '--rnn_size', type=int, required=True,
        help="The RNN size.")
    parser.add_argument(
        '--max_epoch', type=int, default=20,
        help="The max epoch.")
    parser.add_argument(
        '--run', type=int, required=True,
        help="The run number.")
    parser.add_argument(
        '--resume_training', type=bool, required=True,
        help="Boolean, whether to resume training from checkpoint. Pass '' for False.")
    parser.add_argument(
        '--gpu', type=str, required=True,
        help='The gpu number.')
    
    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    overwrite = args.resume_training
    lang_model = args.lang_model
    run = args.run
    
    data_root = pjoin(root, '3_Datasets', 'MSCOCO_captions')
    log_root = pjoin(root, '1_TF_files', 'caption_baseN', 'mscoco_v4')
    print(log_root)
#    data_root = pjoin(root, '3_Datasets', 'InstaPIC1M', 'data')
#    log_root = pjoin(root, '1_TF_files', 'caption_baseN', 'insta_v4')
    log_name = 'v4_0_{}_run_{:02d}'.format(args.name, run)
    
    checkpoint_path = pjoin(log_root, log_name)
    checkpoint_path = None
    #cnn_checkpoint_path = pjoin(cnn_root, 'vgg_16.ckpt') 
    cnn_checkpoint_path = pjoin(cnn_root, 'inception_v1.ckpt')
    #cnn_checkpoint_path = pjoin(cnn_root, 'inception_v4.ckpt')
    
    ###
    
    if lang_model == 'char':
        file_suffix = 's100_include_restval_split'
        itow_file = 'captions/coco_LEGACY_char_itow_%s.json' % file_suffix
        wtoi_file = 'captions/coco_LEGACY_char_wtoi_%s.json' % file_suffix
        caption_file = 'captions/coco_LEGACY_char_%s.h5' % (file_suffix)
        base = None
    elif lang_model == 'bpe':
        file_suffix = 'w0_s24_include_restval_split'
        itow_file = 'captions/coco_LEGACY_bpe4075_base64_itow_%s.json' % file_suffix
        wtoi_file = 'captions/coco_LEGACY_bpe4075_base64_wtoi_%s.json' % file_suffix
        caption_file = 'captions/coco_LEGACY_bpe4075_base64_%s.h5' % (file_suffix)
        base = 64
    else:
        file_suffix = 'w5_s20_include_restval_split'
        itow_file = 'captions/coco_LEGACY_itow_%s.json' % file_suffix
        wtoi_file = 'captions/coco_LEGACY_wtoi_%s.json' % file_suffix
        if lang_model == 'baseN':
            caption_file = 'captions/coco_LEGACY_base%d_%s.h5' % (args.base, file_suffix)
        elif lang_model == 'word':
            caption_file = 'captions/coco_LEGACY_%s.h5' % (file_suffix)
            base = None
#    
#    if lang_model == 'char':
#        file_suffix = 's100_split'
#        itow_file = 'insta_char_itow_%s.json' % file_suffix
#        wtoi_file = 'insta_char_wtoi_%s.json' % file_suffix
#        caption_file = 'insta_char_%s.h5' % (file_suffix)
#        base = None
#    else:
#        file_suffix = 'w5_s18_split'
#        itow_file = 'insta_itow_%s.json' % file_suffix
#        wtoi_file = 'insta_wtoi_%s.json' % file_suffix
#        if lang_model == 'baseN':
#            caption_file = 'insta_base%d_%s.h5' % (args.base, file_suffix)
#        elif lang_model == 'word':
#            caption_file = 'insta_%s.h5' % (file_suffix)
#            base = None
    
    if run == 1:
        initialiser = 'xavier'
        rand_seed = 48964896
    elif run == 2:
        initialiser = 'xavier'
        rand_seed = 88888888
    elif run == 3:
        initialiser = None
        rand_seed = 123456789
    
    if args.fm_projection == 'none':
        args.fm_projection = None
    
    kwargs = dict(
        distort_images = False,
        train_image_model = False,
        train_lang_model = True,
        initialiser = initialiser,
        
        lang_model = lang_model,
        fm_projection = args.fm_projection,
        num_heads = args.num_heads,
        embedding_weight_tying = False,
        multi_softmax = False,
        image_model = args.image_model,
        #vgg_endpoint = 'fc7',
        conv_fm = 'Mixed_4f',                     # InceptionV1
        #conv_fm = 'Mixed_7c',                     # InceptionV3
        #conv_fm = 'Mixed_6h',                   # InceptionV4
        image_embed_size = 1024,
        
        base = args.base,
        rnn = 'LSTM',
        decoder_rnn_size = args.rnn_size,
        num_layers = 1,
        deep_output_layer = False,
        max_caption_length = 40,
        beam_size = 3,
        length_penalty_weight = 0.0,              # TODO: add length_penalty_weight to infer_script
        word_size = args.word_size,
        
        weight_decay = 1e-5,
        dropout_i = 0.35,
        dropout_o = 0.35,
        dropout_im = 0.5,   # not applied
        lr_start = 1e-3,
        lr_end = 1e-3 / 5,
        attention_map_loss_scale = 1.0,
        batch_size = 32,
        max_epoch = args.max_epoch,
        
        num_saves_per_epoch = 2,
        num_logs_per_epoch = 200,
        reduce_lr_every_n_epochs = 4,
        batch_threads = [1, 1, 1],
        capacity_mul_factor = 2.0,
        per_process_gpu_memory_fraction = None,     # was 0.80
        
        rand_seed = rand_seed,
        add_vars_summary = False,
        add_grad_summary = False,
        resume_training = args.resume_training,
        note = note
        )
    
    data_paths = dict(
        data_root = data_root,
        caption_file = caption_file,
        itow_file = itow_file,
        wtoi_file = wtoi_file,
        log_path = pjoin(log_root, log_name),
        save_path = pjoin(log_root, log_name, 'model'),
        cnn_checkpoint_path = cnn_checkpoint_path,
        checkpoint_path = checkpoint_path,
        dropbox_log_path = dropbox_log_path
        )
    
    ###
    
    train.try_to_train(
            try_block = False,
            overwrite = overwrite,
            data_paths = data_paths,
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




