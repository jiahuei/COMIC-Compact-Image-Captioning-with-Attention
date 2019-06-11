#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:02:06 2019

@author: jiahuei

Adapted from `https://github.com/ruotianluo/self-critical.pytorch/blob/master/scripts/prepro_ngrams.py`

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os, json, argparse, time
from six.moves import cPickle as pickle
from collections import defaultdict
from tqdm import tqdm
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
pjoin = os.path.join


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1,n+1):
        for i in xrange(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in tqdm(refs, ncols=100, desc='create_crefs'):
    # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    From `cider_scorer.py` in `coco_caption`.
    '''
    document_frequency = defaultdict(float)
    for refs in tqdm(crefs, ncols=100, desc='compute_doc_freq'):
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def get_ngrams(refs_words, wtoi, params):
    """
    Calculates the n-grams and lengths 
    """
    #refs_idxs = []
    #for ref_words in tqdm(refs_words, ncols=100, desc='Token-to-idx'):
    #    # `ref_words` is a list of captions for an image
    #    ref_idxs = []
    #    for caption in ref_words:
    #        tokens = caption.split(' ')
    #        idx = [wtoi.get(t, wtoi['<UNK>']) for t in tokens]
    #        idx = ' '.join([str(i) for i in idx])
    #        ref_idxs.append(idx)
    #    refs_idxs.append(ref_idxs)
    
    print('\nINFO: Computing term frequency: word.')
    time.sleep(0.1)
    ngram_words = compute_doc_freq(create_crefs(refs_words))
    print('\nINFO: Computing term frequency: indices. (SKIPPED)')
    #time.sleep(0.1)
    #ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    ngram_idxs = None
    return ngram_words, ngram_idxs, len(refs_words)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname((os.path.dirname(CURR_DIR)))
    
    parser.add_argument(
        '--dataset_dir', type=str, default='',
        help='The dataset directory.')
    parser.add_argument(
        '--dataset_file_pattern', type=str,
        default='mscoco_{}_w5_s20_include_restval',
        help='The dataset file pattern, example: `mscoco_{}_w5_s20`.')
    parser.add_argument(
        '--split', type=str, default='train',
        help='The split for generating n-grams.')
    args = parser.parse_args()
    
    dataset = args.dataset_file_pattern.split('_')[0]
    if args.dataset_dir == '':
        args.dataset_dir = pjoin(base_dir, 'datasets', dataset)
    
    # Data format: filepath,w0 w1 w2 w3 w4 ... wN
    fp = pjoin(args.dataset_dir, 'captions',
               args.dataset_file_pattern.format(args.split))
    with open(fp + '.txt', 'r') as f:
        data = [l.strip().split(',') for l in f.readlines()]
    data_dict = {}
    for d in data:
        if d[0] not in data_dict:
            data_dict[d[0]] = []
        data_dict[d[0]].append(d[1].replace('<GO> ', ''))
    
    captions_group = [v for v in data_dict.values()]
    assert len(data_dict.keys()) == len(captions_group)
    
    fp = pjoin(args.dataset_dir, 'captions',
               args.dataset_file_pattern.format('wtoi'))
    with open(fp + '.json', 'r') as f:
        wtoi = json.load(f)
    
    print('\nINFO: Data reading complete.')
    #time.sleep(0.2)
    ngram_words, ngram_idxs, ref_len = get_ngrams(captions_group, wtoi, args)
    
    time.sleep(0.2)
    print('\nINFO: Saving output files.')
    
    fp = pjoin(args.dataset_dir, 'captions', args.dataset_file_pattern)
    with open(fp.format('scst-words') + '.p', 'w') as f:
        pickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len},
                    f, pickle.HIGHEST_PROTOCOL)
    #with open(fp.format('scst-idxs') + '.p', 'w') as f:
    #    pickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len},
    #                f, pickle.HIGHEST_PROTOCOL)
    
    print('\nINFO: Completed.')


