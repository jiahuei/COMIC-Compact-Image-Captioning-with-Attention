#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:53:54 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, json
import prepro_base_v5 as prepro


retokenise = False
include_restval = True
base = 128
word_count_thres = 5
caption_len_thres = 20
train_splits = [10, 12, 14]
#train_splits = None
pad_value = -1
output_prefix = 'coco_LEGACY_base%d' % base
#output_prefix = None
dataset = 'dataset_coco.json'
wtoi_file = 'coco_LEGACY_wtoi_w5_s20_include_restval_split.json'
#wtoi_file = None
itow_file = 'coco_LEGACY_itow_w5_s20_include_restval_split.json'

PATH = '/home/jiahuei/Documents/3_Datasets/MSCOCO_captions/captions'
dataset = os.path.join(PATH, dataset)


### Read the raw JSON file ###

with open(dataset, 'r') as f:
    dataset_coco = json.load(f)

### Tokenise captions ###

tokenised_coco = prepro.tokenise(dataset_coco,
                                 image_id_key='imgid',
                                 retokenise=retokenise)

### Build vocabulary ###

build_vocab = wtoi_file is None or itow_file is None
if build_vocab:
    wtoi, itow = prepro.build_vocab(tokenised_coco,
                                    word_count_thres,
                                    caption_len_thres,
                                    vocab_size=None,
                                    include_restval=include_restval,
                                    pad_value=pad_value,
                                    include_GO_EOS_tokens=False)
else:
    print("INFO: Reusing provided vocabulary.\n")
    with open(os.path.join(PATH, wtoi_file), 'r') as f:
        wtoi = json.load(f)
    with open(os.path.join(PATH, itow_file), 'r') as f:
        itow = json.load(f)

### Convert tokenised words to ids ###

train, valid, test = prepro.tokenised_word_to_baseN_ids(tokenised_coco,
                                                        wtoi,
                                                        caption_len_thres,
                                                        base,
                                                        include_restval)

### Maybe split training data into 4 chunks ###

split_train_data = train_splits is not None
if split_train_data:
    max_word_len = len(prepro.number_to_base(len(wtoi), base))
    train_splits = [t * max_word_len for t in train_splits]
    train = prepro.split_training_data(train, train_splits)

### Output files ###

if output_prefix is not None:
    max_value = base + 1
    if max_value < 127:
        output_dtype = 'int8'
    elif max_value < 32767:
        output_dtype = 'int16'
    else:
        output_dtype = 'int32'
    
    suffix = []
    suffix.append('w%d_s%d' % (word_count_thres, caption_len_thres))
    if include_restval:
        suffix.append('include_restval')
    if split_train_data:
        suffix.append('split')
    if retokenise:
        suffix.append('retokenised')
    suffix = '_'.join(suffix)
    
    prepro.output_files(train, valid, test,
                        wtoi, itow,
                        PATH, output_prefix, suffix,
                        output_dtype,
                        split_train_data,
                        build_vocab)






