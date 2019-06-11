#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 18:34:56 2017

@author: jiahuei

V8
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, json, argparse
import prepro_base as prepro
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURR_DIR, '..', '..', 'common'))
import utils
pjoin = os.path.join


#wtoi_file = 'coco_wtoi_w5_s20_include_restval.json'
#itow_file = 'coco_itow_w5_s20_include_restval.json'


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--dataset_dir', type=str, default='')
    parser.add_argument(
        '--output_prefix', type=str, default='mscoco')
    parser.add_argument(
        '--retokenise', type=bool, default=False)
    parser.add_argument(
        '--include_restval', type=bool, default=True)
    parser.add_argument(
        '--word_count_thres', type=int, default=5)
    parser.add_argument(
        '--caption_len_thres', type=int, default=20)
    parser.add_argument(
        '--pad_value', type=int, default=-1)
    parser.add_argument(
        '--wtoi_file', type=str, default=None)
    parser.add_argument(
        '--itow_file', type=str, default=None)
    
    return parser

if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    
    if args.dataset_dir == '':
        dset_dir = pjoin(os.path.dirname(CURR_DIR), 'mscoco')
    else:
        dset_dir = args.dataset_dir
    out_path = pjoin(dset_dir, 'captions')
    json_path = pjoin(dset_dir, 'dataset_coco.json')
    
    ### Get the caption JSON files ###
    if os.path.isfile(json_path):
        print('INFO: Found file: `dataset_coco.json`')
    else:
        zip_path = utils.maybe_download_from_url(
            r'https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip',
            dset_dir)
        utils.extract_zip(zip_path)
        os.remove(zip_path)
    
    #if os.path.isfile(pjoin(dset_dir, 'annotations', 'captions_val2014.json')):
    #    print('INFO: Found file: `captions_val2014.json`')
    #else:
    #    zip_path = utils.maybe_download_from_url(
    #        r'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
    #        dset_dir)
    #    utils.extract_zip(zip_path)
    #    os.remove(zip_path)
    
    ### Read the raw JSON file ###
    
    with open(json_path, 'r') as f:
        dataset_coco = json.load(f)
    
    ### Tokenise captions ###
    
    tokenised_coco = prepro.tokenise(dataset_coco,
                                     image_id_key='imgid',
                                     retokenise=args.retokenise)
    
    ### Build vocabulary ###
    
    build_vocab = args.wtoi_file is None or args.itow_file is None
    if build_vocab:
        wtoi, itow = prepro.build_vocab(tokenised_coco,
                                        args.word_count_thres,
                                        args.caption_len_thres,
                                        vocab_size=None,
                                        include_restval=args.include_restval,
                                        pad_value=args.pad_value,
                                        include_GO_EOS_tokens=True)
    else:
        print('INFO: Reusing provided vocabulary.\n')
        with open(os.path.join(out_path, args.wtoi_file), 'r') as f:
            wtoi = json.load(f)
        with open(os.path.join(out_path, args.itow_file), 'r') as f:
            itow = json.load(f)
    
    ### Convert tokenised words to text files ###
    
    tokenised_coco = prepro.tokenised_word_to_txt_V1(tokenised_coco,
                                                     args.caption_len_thres,
                                                     args.include_restval)
    
    print('\nINFO: Example captions:')
    for j in range(5):
        print(tokenised_coco['train'][j])
    print('\n')
    
    ### Output files ###
    
    if args.output_prefix is not None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        suffix = []
        suffix.append('w{:d}_s{:d}'.format(
                args.word_count_thres, args.caption_len_thres))
        if args.include_restval:
            suffix.append('include_restval')
        if args.retokenise:
            suffix.append('retokenised')
        suffix = '_'.join(suffix)
        
        for split in tokenised_coco.keys():
            filename = '{}_{}_{}.txt'.format(args.output_prefix, split, suffix)
            with open(pjoin(out_path, filename), 'w') as f:
                f.write('\r\n'.join(tokenised_coco[split]))
        
        # Assert no overlaps between sets
        train_set = set([s.split(',')[0] for s in tokenised_coco['train']])
        valid_set = set([s.split(',')[0] for s in tokenised_coco['valid']])
        test_set = set([s.split(',')[0] for s in tokenised_coco['test']])
        assert not bool(train_set.intersection(valid_set))
        assert not bool(train_set.intersection(test_set))
        assert not bool(valid_set.intersection(test_set))
        
        # Write validation file list
        with open(pjoin(out_path, 'filenames_valid.txt'), 'w') as f:
            f.write('\r\n'.join(list(valid_set)))
        
        # Write test file list
        with open(pjoin(out_path, 'filenames_test.txt'), 'w') as f:
            f.write('\r\n'.join(list(test_set)))
        
        if build_vocab:
            with open('%s/%s_wtoi_%s.json' %
                      (out_path, args.output_prefix, suffix), 'w') as f:
                json.dump(wtoi, f)
            with open('%s/%s_itow_%s.json' %
                      (out_path, args.output_prefix, suffix), 'w') as f:
                json.dump(itow, f)
        
        print('INFO: Saved output text files.\n')
    
    
    ### Get the image files ###
    print('INFO: Checking image files.')
    img_all = train_set.union(valid_set).union(test_set)
    tpath = pjoin(dset_dir, 'train2014')
    vpath = pjoin(dset_dir, 'val2014')
    ext = exv = []
    if os.path.exists(tpath):
        ext = os.listdir(tpath)
        ext = [pjoin('train2014', i) for i in ext]
    if os.path.exists(vpath):
        exv = os.listdir(vpath)
        exv = [pjoin('val2014', i) for i in exv]
    ex = set(ext + exv)
    img_exists = len(ex.intersection(img_all)) == len(img_all)
    
    if img_exists:
        print('INFO: Found exising image files.')
    else:
        zip_path = utils.maybe_download_from_url(
                r'http://images.cocodataset.org/zips/train2014.zip',
                dset_dir)
        utils.extract_zip(zip_path)
        os.remove(zip_path)
        zip_path = utils.maybe_download_from_url(
                r'http://images.cocodataset.org/zips/val2014.zip',
                dset_dir)
        utils.extract_zip(zip_path)
        os.remove(zip_path)
        zip_path = utils.maybe_download_from_url(
                r'http://images.cocodataset.org/zips/test2014.zip',
                dset_dir)
        utils.extract_zip(zip_path)
        os.remove(zip_path)
