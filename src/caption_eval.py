#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:51:43 2017

@author: jiahuei
"""
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')


def evaluate(annFile, resFile):
    # create coco object and cocoRes object
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    
    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)
    
    # evaluate on a subset of images
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    
    # evaluate results
    cocoEval.evaluate()
    
    results = {}
    for metric, score in cocoEval.eval.items():
        #print '%s: %.3f' % (metric, score)
        results[metric] = score
    results['evalImgs'] = cocoEval.evalImgs
    return results


