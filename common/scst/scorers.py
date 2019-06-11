#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:57:43 2019

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys, os
import numpy as np
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
COMMON = os.path.dirname(CURR_DIR)
sys.path.append(os.path.join(COMMON, 'coco_caption'))
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
sys.path.append(os.path.join(COMMON, 'scst', 'cider_ruotianluo'))
from pyciderevalcap.ciderD.ciderD import CiderD
from pyciderevalcap.cider.cider import Cider
#from scst.cider_ruotianluo.pyciderevalcap.ciderD.ciderD import CiderD


_DEBUG = False


class captionScorer(object):
    """
    An object that encapsulates the different scorers to provide a unified
    interface.
    """
    
    def __init__(self, path_to_cached_tokens, metric_weights):
        self._scorer = dict(
                ciderD = CiderD(df=path_to_cached_tokens),
                cider = Cider(df=path_to_cached_tokens),
                bleu = BleuSilent(4))
        self.weights = metric_weights
    
    
    def get_hypo_scores(self, refs, sample, greedy, best_hypo_only=False):
        """
        Length of `sample` can be a multiple of `greedy`.
        Length of `greedy` must match `refs`.
        
        Structure of `sample` must be:
            [[im0_hypo0], ..., [imN_hypo0], [im0_hypo1], ..., [imN_hypo1]]
        Args:
            metric: 
            gts: dict with key <image_id>, value <tokenized ref sentence>
            res: dict with key <image_id>, value <tokenized hypo / candidate sentence>
        """
        # Convert refs, sampled, greedy into appropriate format
        assert isinstance(refs, list)
        assert isinstance(sample, list)
        assert isinstance(greedy, list)
        assert isinstance(refs[0], list)
        assert isinstance(sample[0], list)
        assert isinstance(greedy[0], list)
        assert len(refs) == len(greedy)
        assert len(sample) % len(greedy) == 0
        num_sample = len(sample)
        num_greedy = len(greedy)
        multiple = num_sample // num_greedy
        weights = self.weights
        
        # The key will be of sequence [greedy, sampled]
        # The dictionaries will be of length `num_sample + num_greedy`
        gts = {}
        res = {}
        for idx in range(num_sample):
            if idx < num_greedy:
                res[idx] = greedy[idx]
                gts[idx] = refs[idx]
            res[idx + num_greedy] = sample[idx]
            gts[idx + num_greedy] = refs[idx % num_greedy]
        
        ### DEBUG ###
        if _DEBUG:
            for r in refs:
                for rr in r:
                    assert '\n' not in rr
            out = '\r\n'
            out += 'refs : \r\n{}\r\n\r\n'.format('\r\n'.join([','.join(r) for r in refs]))
            out += 'sample : \r\n{}\r\n\r\n'.format('\r\n'.join([','.join(r) for r in sample]))
            out += 'greedy : \r\n{}\r\n\r\n'.format('\r\n'.join([','.join(r) for r in greedy]))
            out += 'gts : \r\n'
            for k in gts.keys():
                out += '{} --- {}\r\n'.format(k, gts[k])
            out += '\r\n\r\n'
            out += 'res : \r\n'
            for k in res.keys():
                out += '{} --- {}\r\n'.format(k, res[k])
        
        
        # `gts` is a dict with key <int>, 
        #       value <length-N list of tokenised ref sentences>
        # `res` is a dict with key <int>, 
        #       value <length-1 list of tokenised hypo / candidate sentences>
        # `rewards` is [r(sampled) - r(greedy)]
        #metrics = ['cider', 'bleu']
        scores = {}
        #for m in metrics:
        for m in self._scorer:
            if m in weights and np.amax(weights[m]) > 0:
                _, sc = self._scorer[m].compute_score(gts, res)
                if type(weights[m]) == list:
                    for i in range(len(weights[m])):
                        scores['{}_{}'.format(m, i)] = np.array(sc[i]) * weights[m][i]
                else:
                    sc *= weights[m]
                    scores[m] = sc
        
        ### DEBUG ###
        if _DEBUG:
            out += '\r\n\r\n'
            out += 'scores : \r\n'
            for k in scores.keys():
                out += '{} --- {}\r\n'.format(k, scores[k])
            out += '\r\n\r\n'
        
        scores = sum(scores.values())
        sc_greedy = scores[:num_greedy]
        sc_sample = scores[num_greedy:]   # May be longer than num_greedy
        #if num_sample > num_greedy:
        #    multiple = num_sample % num_greedy
        #    sc_greedy = np.stack([sc_greedy] * multiple, axis=0)
        
        
        ### DEBUG ###
        if _DEBUG:
            out += '\r\n\r\n'
            out += 'sc_greedy : \r\n{}\r\n\r\n'.format(sc_greedy)
        
        
        # Maybe select best hypothesis
        if num_sample > num_greedy and best_hypo_only:
            # `sample_beam` will be a list of length `multiple`
            #sample_beam = []
            #for idx in range(multiple):
            #    sample_beam.append(sample[idx*num_greedy : (idx+1)*num_greedy])
            # reshape `sc_sample` into (multiple, num_greedy)
            # final`sc_sample` will be of shape (num_greedy)
            # `best_beam` will be of shape (num_greedy)
            sc_sample = np.reshape(sc_sample, [multiple, num_greedy])
            best_beam = np.argmax(sc_sample, axis=0)
            
            ### DEBUG ###
            if _DEBUG:
                out += 'sc_sample : \r\n{}\r\n\r\n'.format(sc_sample)
                out += 'best_beam : \r\n{}\r\n\r\n'.format(best_beam)
            
            final_hypo = []
            for idx in range(num_greedy):
                offset = num_greedy * best_beam[idx]
                final_hypo.append(sample[idx + offset])
            sc_sample = np.amax(sc_sample, axis=0)
        else:
            if num_sample > num_greedy:
                sc_greedy = np.concatenate([sc_greedy] * multiple)
            final_hypo = sample
        
        if _DEBUG:
            out += 'final_hypo : \r\n{}\r\n\r\n'.format(final_hypo)
            with open('/home/jiahuei/Documents/1_TF_files/caption_M3L/mscoco_v3/scorer.txt', 'a') as f:
                f.write(out + '\r\n\r\n===============================\r\n\r\n')
        
        #rewards = sc_sample - sc_greedy
        return final_hypo, sc_sample, sc_greedy


class BleuSilent(Bleu):
    def compute_score(self, gts, res):
        
        assert(sorted(gts.keys()) == sorted(res.keys()))
        #imgIds = sorted(gts.keys())
        
        bleu_scorer = BleuScorer(n=self._n)
        for id in gts:
            hypo = res[id]
            ref = gts[id]
            
            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)
            
            bleu_scorer += (hypo[0], ref)
        
        # Reduce verbosity
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        
        # return (bleu, bleu_info)
        return score, scores




    
"""

Cross-entropy loss derivative is p_i - y_i,
    where p is the output of softmax and y is the one-hot label.
    This means XE loss grad is prob of class i minus 1.0 if true or 0 if false.

SCST loss derivative is 
    [r(sampled) - r(greedy)] * [p(sample @ t) - oneHot(sample @ t)]
    This means it is equivalent to a weighted version of XE loss, where
    the labels are sampled captions, and the weights are baselined rewards.
        dec_log_ppl = tf.contrib.seq2seq.sequence_loss(
                                        logits=sampled_logits,
                                        targets=sampled_onehot,
                                        weights=sampled_masks,
                                        average_across_batch=False)
        dec_log_ppl = tf.reduce_mean(dec_log_ppl * rewards)


"""














