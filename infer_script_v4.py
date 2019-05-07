#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:53:58 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, json, argparse
import infer_fn_v4 as infer
from utility_functions.captions import infer_ops_v0 as infer_ops
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
from coco_caption_master_spice import caption_eval
input_man = infer.input_man

### Edit these paths ###


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="lol")
    
    parser.add_argument(
        '--log_name', type=str, required=True,
        help="The logging name.")
    
    parser.add_argument(
        '--beam_size', type=int, required=True,
        help="The beam size.")
    
    parser.add_argument(
        '--batch_size', type=int, required=True,
        help="The batch size.")
    
    parser.add_argument(
        '--checkpoints', type=int, nargs='+', required=True,
        help="The checkpoints.")
    
    parser.add_argument(
        '--gpu', type=str, required=True,
        help='The gpu number.')
    
    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    overwrite = True
    beam_size = args.beam_size
    length_penalty_weight = 0.0
    batch_size = args.batch_size
    run_beam_search = True
    get_metric_score = True
    per_process_gpu_memory_fraction = 0.70
    
    root = '/ext_hdd'
    root = '/home/jiahuei/Documents'
    #root = '/home/chun/jh'
    
    data_root = os.path.join(root, '3_Datasets', 'MSCOCO_captions')
    log_root = os.path.join(root, '1_TF_files', 'caption_baseN', 'mscoco_v4')
    log_name = args.log_name
    eval_name = '{}_eval_beam_{}_batch_{}'.format(log_name, beam_size, batch_size)
    
    test_filename_list = os.path.join(data_root, 'filenames_test.txt')
    annotations_file = 'annotations/captions_val2014.json'
    checkpoint_prefix = 'model-'
    tboard_json_prefix = 'run-%s-tag-loss-perplexity_' % log_name
    
    checkpoint_files = args.checkpoints
    
    ###############################################################################
    
    with open(test_filename_list, 'r') as ff:
        filenames = ff.readlines()
    test_filepaths = [os.path.join(data_root, f.strip()) for f in filenames]
    log_path = os.path.join(log_root, log_name)
    eval_log_path = os.path.join(log_root, eval_name)
    vloss_json_path = os.path.join(log_path, '%svalid.json' % tboard_json_prefix)
    tloss_json_path = os.path.join(log_path, '%stest.json' % tboard_json_prefix)
    config_filepath = os.path.join(log_path, 'config.pkl')
    checkpoint_root = os.path.join(log_path, checkpoint_prefix)
    
    if run_beam_search:
        # Load configuration object
        config = input_man.load_config(config_filepath)
        config.data_root = data_root
        config.beam_size = beam_size
        config.length_penalty_weight = length_penalty_weight
        config.batch_size = batch_size
        config.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        if not hasattr(config, 'num_heads'):
            config.num_heads = 1
    else:
        config = None
    
    if os.path.exists(eval_log_path): 
        print("\nINFO: `eval_log_path` already exists.")
    else: 
        print("\nINFO: `eval_log_path` will be created.")
        os.mkdir(eval_log_path)
    
    infer_config = infer_ops.InferenceConfig(
                                config=config,
                                eval_log_path=eval_log_path,
                                checkpoint_root=checkpoint_root,
                                test_filepaths=test_filepaths,
                                annotations_file=annotations_file)
    
    # Maybe load validation and test JSON files
    valid_ppl_dict = None
    test_ppl_dict = None
    if os.path.isfile(vloss_json_path):
        with open(vloss_json_path, 'r') as f:
            json_file = json.load(f)
        valid_ppl_dict = dict([v[1:] for v in json_file])
    
    if os.path.isfile(tloss_json_path):
        with open(tloss_json_path, 'r') as f:
            json_file = json.load(f)
        test_ppl_dict = dict([v[1:] for v in json_file])
    
    
    scores_combined = {}
    for checkpoint in checkpoint_files:
        infer_ops.evaluate_model(
                            infer.run_inference,
                            caption_eval,
                            infer_config,
                            checkpoint,
                            scores_combined,
                            valid_ppl_dict=valid_ppl_dict,
                            test_ppl_dict=test_ppl_dict,
                            run_beam_search=run_beam_search,
                            get_metric_score=get_metric_score)
    
    
    
    
    
    """
    checkpoint_files = [8855,
                        17711,
                        26565,
                        35422,
                        44275,
                        53133,
                        61985,
                        70844,
                        79695,
                        88555,          # epoch 5
                        97405,
                        106266,
                        115115,
                        123977,
                        132825,
                        141688,
                        150535,
                        159399,
                        168245,
                        177110,         # epoch 10
                        185955,
                        194821,
                        203665,
                        212532,         # epoch 12
                        221375,
                        230243,
                        239085,
                        247954,
                        256795,
                        265665,         # epoch 15
                        274505,
                        283376,
                        292215,
                        301087,
                        309925,
                        318798,
                        327635,
                        336509,
                        345345,
                        354220,         # epoch 20
                        363055,
                        371931,
                        380765,
                        389642,
                        398475,
                        407353,
                        416185,
                        425064,
                        433895,
                        442775,         # epoch 25
                        451605,
                        460486,
                        469315,
                        478197,
                        487025,
                        495908,
                        504735,
                        513619,
                        522445,
                        531330]         # epoch 30
    
    """
    
    
