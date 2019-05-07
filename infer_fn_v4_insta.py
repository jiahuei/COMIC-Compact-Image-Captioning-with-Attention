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
import numpy as np
import cPickle as pickle
import os, re, time, json
import model_v4_0 as model
from utility_functions.captions import input_caption_baseN_v2 as input_man


def _number_to_baseN(n, base):
    """Convert any base-10 integer to base-N."""
    if base < 2:
        raise ValueError("Base cannot be less than 2.")
    if n < 0:
        sign = -1
        n *= sign
    elif n == 0:
        return [0]
    else:
        sign = 1
    digits = []
    while n:
        digits.append(sign * int(n % base))
        n //= base
    return digits[::-1]


def _baseN_arr_to_dec(baseN_array, base):
    """Convert base-N array / list to base-10 number."""
    result = 0
    power = len(baseN_array) - 1
    for num in baseN_array:
        result += num * pow(base, power)
        power -= 1
    return result


def _baseN_id_to_caption(ids, config):
    """Convert base-N word IDs to words / sentence."""
    captions = []
    base = config.base
    vocab_size = len(config.itow)
    word_len = len(_number_to_baseN(vocab_size, base))
    for i in range(ids.shape[0]):
        caption = []
        row = [wid for wid in ids[i, :] if wid < base and wid >= 0]
        if len(row) % word_len != 0:
            row = row[:-1]
        for j in range(0, len(row), word_len):
            word_id = _baseN_arr_to_dec(row[j:j + word_len], base)
            if word_id < vocab_size:
                caption.append(config.itow[str(word_id)])
            else:
                pass
        captions.append(' '.join(caption))
    return captions


def _word_id_to_caption(ids, config):
    """Convert word IDs to words / sentence."""
    captions = []
    for i in range(ids.shape[0]):
        caption = []
        row = [wid for wid in ids[i, :] 
                if wid >= 0 and wid != config.wtoi['<EOS>']]
        for j in range(len(row)):
            caption.append(config.itow[str(row[j])])
        captions.append(' '.join(caption))
    return captions


def _char_id_to_caption(ids, config):
    """Convert character IDs to words / sentence."""
    captions = []
    for i in range(ids.shape[0]):
        caption = []
        row = [wid for wid in ids[i, :] 
                if wid >= 0 and wid != config.wtoi['<EOS>']]
        for j in range(len(row)):
            caption.append(config.itow[str(row[j])])
        captions.append(''.join(caption))
    return captions


def _bpe_id_to_caption(ids, config):
    """Convert BPE encoded word IDs to words / sentence."""
    captions = _baseN_id_to_caption(ids, config)
    captions_processed = []
    for c in captions:
        captions_processed.append(c.replace('@@ ', ''))
    return captions


def run_inference(infer_config, checkpoint_string):
    """
    Main inference function. Builds and executes the model.
    """
    
    ckpt_path = infer_config.checkpoint_root + checkpoint_string
    ckpt_file = os.path.basename(ckpt_path)
    
    tf.set_random_seed(infer_config.config.rand_seed)
    inputs_man = input_man.InputManagerInfer(infer_config.config,
                                             infer_config.test_filepaths)
    config = inputs_man.config
    
    if config.lang_model == 'baseN':
        _id_to_caption = _baseN_id_to_caption
    elif config.lang_model == 'bpe':
        _id_to_caption = _bpe_id_to_caption
    elif config.lang_model == 'word':
        _id_to_caption = _word_id_to_caption
    elif config.lang_model == 'char':
        _id_to_caption = _char_id_to_caption
    
    # Setup input pipeline & Build model
    g = tf.Graph()
    with g.as_default():
        inputs_man.initialise_tf_queue()
        with tf.name_scope("infer"):
            model_infer = model.Model(
                                config,
                                mode='inference',
                                batch_ops=inputs_man.test_batch_ops, 
                                reuse=False,
                                name='infer')
        init = tf.local_variables_initializer()
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
    g.finalize()
    
    r = config.per_process_gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
    num_batches = int(config.test_size / config.batch_size)
    filenames = inputs_man.test_set_filenames
    
    outputs_dict = {'captions' : {}, 'attention' : {}}
    coco_json = []
    with sess:
        sess.run(init)
        runners = tf.train.start_queue_runners(coord=coord, sess=sess)
        # Restore model from checkpoint
        saver.restore(sess, ckpt_path)
        
        print("INFO: Graph constructed. Starting inference.")
        start_time = time.time()
        
        for step in xrange(num_batches):
            infer_outputs, attn_maps = sess.run(model_infer.infer_output)
            captions = _id_to_caption(infer_outputs, config)
            if config.beam_size == 1:
                attn_maps = np.split(attn_maps, config.batch_size)
            
            # Get image ids, compile results
            batch_start = step * config.batch_size
            batch_end = (step + 1) * config.batch_size
            batch_filenames = filenames[batch_start : batch_end]
            
            for i, f in enumerate(batch_filenames):
                outputs_dict['captions'][f] = captions[i]
                if config.beam_size == 1:
                    outputs_dict['attention'][f] = attn_maps[i]
                coco_dict = {}
                coco_dict['caption'] = unicode(captions[i])
                coco_dict['image_id'] = f
                coco_json.append(coco_dict)
            
            if step % int(num_batches / 20) == 0:
                print("\n>>> %3.2f %%\t(checkpoint: %s)\n" % 
                          (step / num_batches * 100, checkpoint_string))
                print("Example captions:\n%s" % "\n".join(captions[:3]))
        
        t = time.time() - start_time
        
        # Shutdown everything
        coord.request_stop()
        coord.join(runners)
        sess.close()
    
    assert len(filenames) == len(coco_json)
        
    # Dump output files
    outputs_dict['beam_size'] = config.beam_size
    outputs_dict['max_caption_length'] = config.max_caption_length
    out_fname = 'captions_results___%s.json' % ckpt_file
    raw_out_fname = 'outputs_dict___%s.json' % ckpt_file
    #with open(os.path.join(infer_config.eval_log_path, raw_out_fname), 'w') as f:
    #    json.dump(outputs_dict, f)
    with open(os.path.join(infer_config.eval_log_path, out_fname), 'w') as f:
        json.dump(coco_json, f)
    with open(os.path.join(infer_config.eval_log_path, 'output.pkl'), 'wb') as f:
        pickle.dump(outputs_dict, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(infer_config.eval_log_path, 'infer_speed.txt'), 'a') as f:
        f.write('\r\n{}'.format(len(filenames) / t))
    print("\nINFO: Inference completed. Time taken: {:4.2f} mins\n".format(t/60))

