#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:01:49 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

from tqdm import tqdm
import os, time, json, h5py
import numpy as np
pjoin = os.path.join
_ROOT = '/home/jiahuei/Documents'


#data_h5 = '3_Datasets/MSCOCO_captions/captions/coco_LEGACY_w5_s20_include_restval_split.h5'
#vocab_json = '3_Datasets/MSCOCO_captions/captions/coco_LEGACY_itow_w5_s20_include_restval_split.json'
data_h5 = '3_Datasets/InstaPIC1M/captions/insta_LEGACY_v25595_s18_split.h5'
vocab_json = '3_Datasets/InstaPIC1M/captions/insta_LEGACY_itow_v25595_s18_split.json'
caption_json = '1_TF_files/caption_baseN/insta_v4/v4_0_icpV1_lstm_b160_m4f_b32_h8_tied_r512_w256_run_01_eval_beam_3_batch_25/captions_results___model-576042.json'
res_file = '1_TF_files/caption_baseN/caption_statistics.txt'


data_h5 = pjoin(_ROOT, data_h5)
vocab_json = pjoin(_ROOT, vocab_json)
caption_json = pjoin(_ROOT, caption_json)
res_file = pjoin(_ROOT, res_file)

if not os.path.exists(os.path.split(res_file)[0]):
    os.makedirs(os.path.split(res_file)[0])

# Load captions
print('INFO: Loading captions.')
with open(caption_json, 'r') as f:
    captions = json.load(f)
captions_list = [d['caption'] for d in captions]

train = []
with h5py.File(data_h5, 'r') as f:
    for i in range(4):
        train.append(f['train_%d/targets' % i][:])
max_len = train[-1].shape[-1]

# Load vocab
with open(vocab_json, 'r') as f:
    itow = json.load(f)

# Load training data
print('INFO: Loading training data.')
time.sleep(0.2)
train_combined = None
for t in train:
    t = np.pad(t, (0, max_len - t.shape[-1]),
               'constant', constant_values=(0, -1))
    if train_combined is None:
        train_combined = t
    else:
        train_combined = np.concatenate([train_combined, t], axis=0)

train_list = [' '.join([itow[str(i)] for i in r]) \
              .replace('<PAD>', '').replace('<EOS>', '') \
              .strip() for r in tqdm(train_combined)]

# Get statistics
print('\nINFO: Generating statistics.')
time.sleep(0.2)
appear_in_train = 0
average_length = []
for c in tqdm(captions_list):
    if c in train_list:
        appear_in_train += 1
    average_length.append(len(c.split(' ')))

average_length = '{:4.1f}'.format(np.mean(average_length))
percent_unique = (1 - (appear_in_train / len(captions_list))) * 100
percent_unique = '{:4.2f}'.format(percent_unique)
res = 'Average length of captions: {}\r\n'.format(average_length)
res += 'Percentage of unique captions: {} %\r\n\r\n'.format(percent_unique)

print('\n\n' + res)

res = '{}\r\n'.format(caption_json) + res
with open(res_file, 'a') as f:
    f.write(res)




