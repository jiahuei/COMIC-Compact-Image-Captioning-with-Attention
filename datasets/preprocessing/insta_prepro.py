#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:55:32 2017

@author: jiahuei

V8
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, copy
import json, re, random
from tqdm import tqdm
import prepro_base as prepro
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common'))
import utils
pjoin = os.path.join


output_prefix = 'insta'
word_count_thres = 5
vocab_size = 25595
vocab_size = None
caption_len_thres = 15
pad_value = -1
#wtoi_file = 'insta_wtoi_w5_s15_split.json'
#itow_file = 'insta_itow_w5_s15_split.json'
wtoi_file = itow_file = None


dset_dir = pjoin(os.path.split(os.path.dirname(__file__))[0], 'InstaPIC1M')
out_path = pjoin(dset_dir, 'captions')
cap_train_json_path = pjoin(dset_dir, 'json', 'insta-caption-train.json')
cap_test1_json_path = pjoin(dset_dir, 'json', 'insta-caption-test1.json')


### Get the caption JSON files ###
json_exists = (os.path.isfile(cap_train_json_path) and
               os.path.isfile(cap_test1_json_path))
tgz_path = pjoin(dset_dir, 'json.tar.gz')
if json_exists:
    print('INFO: Found exising json files.')
else:
    if os.path.isfile(tgz_path):
        print('INFO: Found file: `json.tar.gz`')
    else:
        utils.maybe_download_from_google_drive(
                            r'0B3xszfcsfVUBdG0tU3BOQWV0a0E',
                            tgz_path,
                            file_size=669*1024**2)
    utils.extract_tar_gz(tgz_path)
    if os.path.isfile(tgz_path):
        os.remove(tgz_path)

# For tokenization
# https://github.com/cesc-park/attend2u/blob/master/scripts/generate_dataset.py
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
    for user_id, posts in tqdm(train_json.items(),
                               ncols=100, desc='Tokenising train data'):
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
    for user_id, posts in tqdm(test1_json.items(),
                               ncols=100, desc='Tokenising test1 data'):
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

print('\nINFO: Reading JSON files.\n')
with open(cap_train_json_path, 'r') as f:
    cap_train_json = json.load(f)
with open(cap_test1_json_path, 'r') as f:
    cap_test1_json = json.load(f)


### Tokenize all ###
tokenised_insta = tokenize_all(cap_train_json, cap_test1_json)
tokenised_insta_copy = copy.deepcopy(tokenised_insta)
print('')


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
    with open(os.path.join(out_path, wtoi_file), 'r') as f:
        wtoi = json.load(f)
    with open(os.path.join(out_path, itow_file), 'r') as f:
        itow = json.load(f)

### Convert tokenised words to text files ###

tokenised_insta = prepro.tokenised_word_to_txt_V1(tokenised_insta,
                                                  caption_len_thres,
                                                  include_restval=False)

print('\nINFO: Example captions:')
for j in range(5):
    print(tokenised_insta['train'][j])
print('\n')

### Output files ###

if output_prefix is not None:
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    if vocab_size is None:
        suffix = 'w{}_s{}'.format(word_count_thres, caption_len_thres)
    else:
        suffix = 'v{}_s{}'.format(vocab_size, caption_len_thres)
    
    for split in tokenised_insta.keys():
        filename = '{}_{}_{}.txt'.format(output_prefix, split, suffix)
        with open(pjoin(out_path, filename), 'w') as f:
            f.write('\r\n'.join(tokenised_insta[split]))
    
    # Assert no overlaps between sets
    train_set = set([s.split(',')[0] for s in tokenised_insta['train']])
    valid_set = set([s.split(',')[0] for s in tokenised_insta['valid']])
    test_set = set([s.split(',')[0] for s in tokenised_insta['test']])
    assert not bool(train_set.intersection(valid_set))
    assert not bool(train_set.intersection(test_set))
    assert not bool(valid_set.intersection(test_set))
    
    # Write train file list
    #with open(pjoin(OUT_PATH, 'filenames_train.txt'), 'w') as f:
    #    f.write('\r\n'.join(list(train_set)))
    
    # Write validation file list
    with open(pjoin(out_path, 'filenames_valid.txt'), 'w') as f:
        f.write('\r\n'.join(list(valid_set)))
    
    # Write test file list
    with open(pjoin(out_path, 'filenames_test.txt'), 'w') as f:
        f.write('\r\n'.join(list(test_set)))
    
    if build_vocab:
        with open('%s/%s_wtoi_%s.json' %
                  (out_path, output_prefix, suffix), 'w') as f:
            json.dump(wtoi, f)
        with open('%s/%s_itow_%s.json' %
                  (out_path, output_prefix, suffix), 'w') as f:
            json.dump(itow, f)
    
    # Generate COCO style annotation file
    test_ann = dict(images=[],
                    info='',
                    type='captions',
                    annotations=[],
                    licenses='')
    
    for d in tokenised_insta_copy:
        if d['split'] not in ['test', 'val']:
            continue
        test_ann['images'].append({'id': d['image_id']})
        test_ann['annotations'].append(
                {'caption': d['raw'][0].replace('_UNK', '<UNK>'),
                 'id': 0,
                 'image_id': d['image_id']})
    
    with open(pjoin(out_path, 'captions_insta_test1.json'), 'w') as f:
        json.dump(test_ann, f)
    
    print('INFO: Saved output text files.\n')


### Get the image files ###
img_all = train_set.union(valid_set).union(test_set)
ex = []
if os.path.exists(pjoin(dset_dir, 'images')):
    ex = os.listdir(pjoin(dset_dir, 'images'))
    ex = [pjoin('images', i) for i in ex]
ex = set(ex)
img_exists = len(ex.intersection(img_all)) == len(img_all)
tgz_path = pjoin(dset_dir, 'images.tar.gz')
if img_exists:
    print('INFO: Found exising image files.')
else:
    if os.path.isfile(tgz_path):
        print('INFO: Found file: `images.tar.gz`')
    else:
        utils.maybe_download_from_google_drive(
                            r'0B3xszfcsfVUBVkZGU2oxYVl6aDA',
                            tgz_path,
                            file_size=20*1024**3)
    utils.extract_tar_gz(tgz_path)
    if os.path.isfile(tgz_path):
        os.remove(tgz_path)

