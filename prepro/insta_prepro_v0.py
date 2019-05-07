#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:55:32 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, json, re, random
from tqdm import tqdm
import prepro_base_v5 as prepro
pjoin = os.path.join


output_prefix = 'insta_LEGACY'
#output_prefix = None
word_count_thres = 5
vocab_size = 25595
#vocab_size = None
caption_len_thres = 18
train_splits = [6, 9, 12]
#train_splits = None
pad_value = -1
wtoi_file = 'insta_LEGACY_wtoi_v25595_s18_split.json'
wtoi_file = None
itow_file = 'insta_LEGACY_itow_v25595_s18_split.json'

PATH = '/home/jiahuei/Documents/3_Datasets/InstaPIC1M'
cap_train_json_path = pjoin(PATH, 'json', 'insta-caption-train.json')
cap_test1_json_path = pjoin(PATH, 'json', 'insta-caption-test1.json')

# For tokenization
try:
  # UCS-4
  EMOTICON = re.compile(u'(([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF]))')
except Exception, e:
  # UCS-2
  EMOTICON = re.compile(u'(([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF]))')
NOT_EMOTICON = re.compile(r'(\\U([0-9A-Fa-f]){8})|(\\u([0-9A-Fa-f]){4})')



def tokenize(sentence):
    """Tokenize a sentence"""
    if isinstance(sentence, list):
        sentence = ' '.join(sentence)

    sentence = sentence.replace('#', ' #')
    sentence = sentence.replace('@', ' @')
    sentence = sentence.replace('\n', ' ')
    sentence = sentence.lower()
    sentence = re.sub(r'@[a-zA-Z0-9._]+', '@username', sentence)    # change username
    sentence = EMOTICON.sub(r'@@byeongchang\1 ', sentence)
    sentence = sentence.encode('unicode-escape')    # for emoticons
    sentence = re.sub(r'@@byeongchang\\', '@@byeongchang', sentence)
    sentence = NOT_EMOTICON.sub(r' ', sentence)
    sentence = re.sub(r"[\-_]", r"-", sentence)    # incoporate - and _
    sentence = re.sub(r"([!?,\.\"])", r" ", sentence)    # remove duplicates on . , ! ?
    sentence = re.sub(r"(?<![a-zA-Z0-9])\-(?![a-zA-Z0-9])", r"", sentence)    # remove - if there is no preceed or following
    sentence = ' '.join(re.split(r'[^a-zA-Z0-9#@\'\-]+', sentence))
    sentence = re.sub(r'@@byeongchang', r' \\', sentence)
    return sentence.split()


def tokenize_all(train_json, test1_json):
    """
    Tokenize sentences in raw dataset

    Args:
        train_json, test1_json: raw json object
        key: 'caption' or 'tags'
    """
    
    #print("\nINFO: Tokenising captions.\n")
    tokenised_data = []
    # Train data
    for user_id, posts in tqdm(train_json.items(), ncols=70, desc="Tokenising train data"):
        for post_id, post in posts.items():
            img_id = '{}_@_{}'.format(user_id, post_id)
            temp_dict = dict(split='train',
                             filepath=pjoin('images', img_id),
                             image_id=img_id,
                             raw=[post['caption']],
                             tokens=[tokenize(post['caption'])])
            tokenised_data.append(temp_dict)
    
    # Validation data
    random.seed(4896)
    random.shuffle(tokenised_data)
    for i in range(2000):
        tokenised_data[i]['split'] = 'val'
    
    # Test1 data
    for user_id, posts in tqdm(test1_json.items(), ncols=70, desc="Tokenising test1 data"):
        for post_id, post in posts.items():
            img_id = '{}_@_{}'.format(user_id, post_id)
            temp_dict = dict(split='test',
                             filepath=pjoin('images', img_id),
                             image_id=img_id,
                             raw=[post['caption']],
                             tokens=[tokenize(post['caption'])])
            tokenised_data.append(temp_dict)
    return tokenised_data


### Read the raw JSON file ###

print("\nINFO: Reading JSON files.\n")
with open(cap_train_json_path, 'r') as f:
    cap_train_json = json.load(f)
with open(cap_test1_json_path, 'r') as f:
    cap_test1_json = json.load(f)


### Tokenize all ###
tokenised_insta = tokenize_all(cap_train_json, cap_test1_json)

import matplotlib.pyplot as plt
from PIL import Image
img = Image.open(pjoin(PATH, tokenised_insta[0]['filepath']))
img_plot = plt.imshow(img)

### Build vocabulary ###

build_vocab = wtoi_file is None or itow_file is None
if build_vocab:
    wtoi, itow = prepro.build_vocab(tokenised_insta,
                                    word_count_thres,
                                    caption_len_thres,
                                    vocab_size=vocab_size,
                                    include_restval=False,
                                    pad_value=pad_value,
                                    include_GO_EOS_tokens=True)
else:
    print("INFO: Reusing provided vocabulary.\n")
    with open(os.path.join(PATH, wtoi_file), 'r') as f:
        wtoi = json.load(f)
    with open(os.path.join(PATH, itow_file), 'r') as f:
        itow = json.load(f)

### Convert tokenised words to ids ###

train, valid, test = prepro.tokenised_word_to_ids(tokenised_insta,
                                                  wtoi,
                                                  caption_len_thres,
                                                  include_restval=False)
print("\nINFO: Example captions:")
for j in range(5):
    print(' '.join([itow[i] for i in test['targets'][j,:] if i > pad_value]))
print("\n")

### Maybe split training data into 4 chunks ###

split_train_data = train_splits is not None
if split_train_data:
    train = prepro.split_training_data(train, train_splits)

### Output files ###

if output_prefix is not None:
    max_value = len(itow)
    if max_value < 127:
        output_dtype = 'int8'
    elif max_value < 32767:
        output_dtype = 'int16'
    else:
        output_dtype = 'int32'
    
    if vocab_size is None:
        suffix = 'w{}_s{}'.format(word_count_thres, caption_len_thres)
    else:
        suffix = 'v{}_s{}'.format(vocab_size, caption_len_thres)
    if split_train_data:
        suffix += '_split'
    
    prepro.output_files(train, valid, test,
                        wtoi, itow,
                        PATH, output_prefix, suffix,
                        output_dtype,
                        split_train_data,
                        build_vocab)
    
    # Generate COCO style annotation file
    test_data = []
    for user_id, posts in tqdm(cap_test1_json.items(), ncols=70, desc="Tokenising test1 data"):
        for post_id, post in posts.items():
            img_id = '{}_@_{}'.format(user_id, post_id)
            temp_dict = dict(split='test',
                             filepath=pjoin('images', img_id),
                             image_id=img_id,
                             raw=[post['caption']],
                             tokens=[tokenize(post['caption'])])
            test_data.append(temp_dict)
    ann = dict(images=[],
               info='',
               type='captions',
               annotations=[],
               licenses='')
    for d in test_data:
        ann['images'].append({'id': d['image_id']})
        ann['annotations'].append(
                {'caption': d['raw'][0].replace('_UNK', '_UNKNOWN'),
                 'id': 0,
                 'image_id': d['image_id']})
    
    with open(pjoin(PATH, 'captions_insta_test1.json'), 'w') as f:
        json.dump(ann, f)
    
    # Write test1 file list
    test_list = list(set([d['filepath'] for d in test_data]))
    with open(pjoin(PATH, 'filenames_test.txt'), 'w') as f:
        f.write('\r\n'.join(test_list))
    


