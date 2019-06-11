#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:12:19 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#import numpy as np
import cPickle as pickle
import os, sys, re, time, json
from tqdm import tqdm
from model import CaptionModel
import inputs.manager_image_caption as inputs
#from coco_caption import caption_eval
import ops

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURR_DIR, '..', 'common', 'coco_caption'))
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.3f')
pjoin = os.path.join


P_COCO = re.compile(r'(?<=_)\d+')
P_CKPT = re.compile(r'\d+')


def _baseN_arr_to_dec(baseN_array, base):
    """Convert base-N array / list to base-10 number."""
    result = 0
    power = len(baseN_array) - 1
    for num in baseN_array:
        result += num * pow(base, power)
        power -= 1
    return result


def id_to_caption(ids, config):
    captions = []
    if config.token_type == 'radix':
        # Convert Radix IDs to sentence.
        base = config.radix_base
        vocab_size = len(config.itow)
        word_len = len(ops.number_to_base(vocab_size, base))
        for i in range(ids.shape[0]):
            sent = []
            row = [wid for wid in ids[i, :] if wid < base and wid >= 0]
            if len(row) % word_len != 0:
                row = row[:-1]
            for j in range(0, len(row), word_len):
                word_id = _baseN_arr_to_dec(row[j:j + word_len], base)
                if word_id < vocab_size:
                    sent.append(config.itow[str(word_id)])
                else:
                    pass
            captions.append(' '.join(sent))
    else:
        # Convert word / char IDs to sentence.
        for i in range(ids.shape[0]):
            row = [wid for wid in ids[i, :] 
                    if wid >= 0 and wid != config.wtoi['<EOS>']]
            sent = [config.itow[str(w)] for w in row]
            if config.token_type == 'word':
                captions.append(' '.join(sent))
            elif config.token_type == 'char':
                captions.append(''.join(sent))
    return captions


def run_inference(config, curr_ckpt_path):
    """
    Main inference function. Builds and executes the model.
    """
    
    ckpt_dir, ckpt_file = os.path.split(curr_ckpt_path)
    ckpt_num = P_CKPT.findall(ckpt_file)[0]             # Checkpoint number
    
    # Setup input pipeline & Build model
    print('TensorFlow version: r{}'.format(tf.__version__))
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(config.rand_seed)
        inputs_man = inputs.InputManager(config, is_inference=True)
        c = inputs_man.config
        batch_size = c.batch_size_infer
        
        with tf.name_scope('infer'):
            m_infer = CaptionModel(
                                c,
                                mode='infer',
                                batch_ops=inputs_man.batch_infer, 
                                reuse=False,
                                name='inference')
        init_fn = tf.local_variables_initializer()
        saver = tf.train.Saver()
    
    filenames = inputs_man.filenames_infer
    r = config.per_process_gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
    num_batches = int(c.split_sizes['infer'] / batch_size)
    
    raw_outputs = dict(captions = {},
                       attention = {},
                       image_ids = {},
                       beam_size = c.infer_beam_size,
                       max_caption_length = c.infer_max_length,
                       checkpoint_path = curr_ckpt_path,
                       checkpoint_number = ckpt_num)
    coco_json = []
    with sess:
        sess.run(init_fn)
        # Restore model from checkpoint
        saver.restore(sess, curr_ckpt_path)
        g.finalize()
        
        print("INFO: Graph constructed. Starting inference.")
        start_time = time.time()
        
        desc = 'Inference: checkpoint {}'.format(ckpt_num)
        for step in tqdm(range(num_batches), desc=desc, ncols=100):
            word_ids, attn_maps = sess.run(m_infer.infer_output)
            captions = id_to_caption(word_ids, c)
            #attn_maps = np.split(attn_maps, batch_size)
            
            # Get image ids, compile results
            batch_start = step * batch_size
            batch_end = (step + 1) * batch_size
            batch_filenames = filenames[batch_start : batch_end]
            
            for i, f in enumerate(batch_filenames):
                image_id = f.replace('.jpg', '')
                image_id = P_COCO.findall(image_id)
                if len(image_id) > 0:
                    image_id = int(image_id[0])
                else:
                    image_id = int(image_id)
                raw_outputs['captions'][f] = captions[i]
                #if c.infer_beam_size == 1:
                raw_outputs['attention'][f] = attn_maps[i]
                raw_outputs['image_ids'][f] = image_id
                coco_json.append(dict(image_id = image_id,
                                      caption = unicode(captions[i])))
        
        print("\nExample captions:\n{}\n".format("\n".join(captions[:3])))
        t = time.time() - start_time
        sess.close()
    
    # Ensure correctness
    assert len(filenames) == len(list(set(filenames)))
    assert len(filenames) == len(coco_json)
    assert len(filenames) == len(raw_outputs['image_ids'].keys())
    
    # Dump output files
    raw_output_fname = 'outputs___{}.pkl'.format(ckpt_num)
    coco_json_fname = 'captions___{}.json'.format(ckpt_num)
    
    # Captions with attention maps
    if c.save_attention_maps:
        with open(pjoin(c.infer_save_path, raw_output_fname), 'wb') as f:
            pickle.dump(raw_outputs, f, pickle.HIGHEST_PROTOCOL)
    # Captions with image ids
    with open(pjoin(c.infer_save_path, coco_json_fname), 'w') as f:
        json.dump(coco_json, f)
    if not os.path.isfile(pjoin(c.infer_save_path, 'infer_speed.txt')):
        out = ['Using GPU #: {}'.format(c.gpu),
               'Inference batch size: {}'.format(c.batch_size_infer),
               'Inference beam size: {}'.format(c.infer_beam_size),
               '']
        with open(pjoin(c.infer_save_path, 'infer_speed.txt'), 'a') as f:
            f.write('\r\n'.join(out))
    with open(pjoin(c.infer_save_path, 'infer_speed.txt'), 'a') as f:
        f.write('\r\n{}'.format(len(filenames) / t))
    print("\nINFO: Inference completed. Time taken: {:4.2f} mins\n".format(t/60))


def evaluate_model(config,
                   curr_ckpt_path,
                   scores_combined,
                   valid_ppl_dict=None,
                   test_ppl_dict=None):
    """
    Evaluates the model and returns the metric scores.
    """
    c = config
    
    ckpt_dir, ckpt_file = os.path.split(curr_ckpt_path)
    ckpt_num = int(P_CKPT.findall(ckpt_file)[0])
    output_filename = 'captions___{}.json'.format(ckpt_num)
    coco_json = pjoin(c.infer_save_path, output_filename)
    
    if c.run_inference:
        if not os.path.isfile('{}.index'.format(curr_ckpt_path)):
            print("WARNING: `{}.index` not found. Checkpoint skipped.".format(ckpt_file))
            return None
        if os.path.isfile(coco_json):
            print("INFO: Found caption file `{}`. Skipping inference.".format(
                        os.path.basename(coco_json)))
        else:
            # Beam search to obtain captions
            run_inference(config, curr_ckpt_path)
    
    if not c.get_metric_score:
        return None
    
    # Evaluate captions
    print("\nINFO: Evaluation: checkpoint \t {}\n" .format(ckpt_num))
    
    #results = caption_eval.evaluate(c.annotations_file, coco_json)
    results = evaluate_captions(c.annotations_file, coco_json)
    
    # Compile scores
    metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
               'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
    
    scores = ['{:1.3f}'.format(results[m]) for m in metrics]
    scores_str = ['{}: {:1.3f}'.format(m, results[m]) for m in metrics]
    scores_combined[ckpt_num] = results
    
    valid_ckpt_missing = valid_ppl_dict is None \
                        or ckpt_num not in valid_ppl_dict.keys()
    test_ckpt_missing = test_ppl_dict is None \
                        or ckpt_num not in test_ppl_dict.keys()
    #valid_ckpt_exists = test_ckpt_exists = False
    score_file = pjoin(c.infer_save_path, 'metric_scores')
    
    # Finally write aggregated scores to file
    with open(score_file + ".txt", 'a') as f:
        out_string = "===================================\r\n"
        out_string += "%s\r\n" % ckpt_file
        out_string += "Beam size: %d\r\n" % c.infer_beam_size
        out_string += "===================================\r\n"
        out_string += "%s\r\n" % "\r\n".join(scores_str)
        out_string += "Perplexity (valid): "
        
        if valid_ckpt_missing:
            out_string += "N/A\r\n"
        else:
            out_string += "%2.3f\r\n" % valid_ppl_dict[ckpt_num]
            
        out_string += "Perplexity (test): "
        if test_ckpt_missing:
            out_string += "N/A\r\n"
        else:
            out_string += "%2.3f\r\n" % test_ppl_dict[ckpt_num]
        out_string += "\r\n\r\n"
        f.write(out_string)
    
    # Write scores to file in CSV style
    with open(score_file + ".csv", 'a') as f:
        out_string = "%d," % ckpt_num
        out_string += "%s," % ",".join(scores)
        if valid_ckpt_missing:
            out_string += "N/A,"
        else:
            out_string += "%2.3f," % valid_ppl_dict[ckpt_num]
        if test_ckpt_missing:
            out_string += "N/A\r\n"
        else:
            out_string += "%2.3f\r\n" % test_ppl_dict[ckpt_num]
        f.write(out_string)
    
    # Write individual scores
    #_dprint("results['evalImgs'] is: {}".format(results['evalImgs']))
    sorted_cider = sorted(results['evalImgs'],
                          key=lambda k: k['CIDEr'],
                          reverse=True)
#    json_file = score_file + "_detailed.json"
#    if os.path.isfile(json_file):
#        with open(json_file, 'r') as f:
#            cider_dict = json.load(f)
#        cider_dict[ckpt_num] = sorted_cider
#    else:
#        cider_dict = {ckpt_num: sorted_cider}
#    with open(json_file, 'w') as f:
#        json.dump(cider_dict, f)
    json_file = score_file + "_detailed_{}.json".format(ckpt_num)
    with open(json_file, 'w') as f:
        json.dump(sorted_cider, f)
    
    return scores_combined


def evaluate_captions(annFile, resFile):
    # create coco object and cocoRes object
    coco = COCO(pjoin(CURR_DIR, annFile))
    cocoRes = coco.loadRes(pjoin(CURR_DIR, resFile))
    
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


