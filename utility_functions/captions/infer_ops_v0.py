#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:06:35 2017

@author: jiahuei
"""
import os, collections, json


class InferenceConfig(
        collections.namedtuple("InferenceConfig",
                               ("config",
                                "eval_log_path",
                                "checkpoint_root",
                                "test_filepaths",
                                "annotations_file"))):
    """
    `namedtuple` storing the state of a `InferenceConfig`.
    
    Contains:
        
        - 
    """
    
    def clone(self, **kwargs):
        """Clone this object, overriding components provided by kwargs."""
        return super(InferenceConfig, self)._replace(**kwargs)


def check_json(json_file, checkpoint_files):
    """
    Check if all the checkpoints are available.
    """
    json_dict = dict([v[1:] for v in json_file])
    if all([x in json_dict.keys() for x in checkpoint_files]):
        return True
    else:
        return False


def evaluate_model(run_inference,
                   caption_eval,
                   infer_config,
                   checkpoint,
                   scores_combined,
                   valid_ppl_dict=None,
                   test_ppl_dict=None,
                   run_beam_search=True,
                   get_metric_score=True):
    """
    Evaluates the model and returns the metric scores.
    """
    ckpt_str = str(checkpoint)
    ckpt_path = infer_config.checkpoint_root + ckpt_str
    ckpt_file = os.path.basename(ckpt_path)
    output_filename = 'captions_results___%s.json' % ckpt_file
    res_file = os.path.join(infer_config.eval_log_path, output_filename)
    
    if run_beam_search:
        if os.path.isfile('%s.index' % ckpt_path) is False:
            print("WARNING: %s not found." % os.path.basename(ckpt_path))
            return None
        if os.path.isfile(res_file) is True:
            print("INFO: Found inference result file `%s`. Skipping." %
                  os.path.basename(res_file))
        else:
            # Beam search to obtain captions
            run_inference(infer_config, ckpt_str)
    
    if not(get_metric_score):
        return None
    
    # Evaluate captions
    print("\nINFO: Evaluating checkpoint:\t %s\n" % ckpt_str)
    
    results = caption_eval.evaluate(infer_config.annotations_file, res_file)
    
    # Compile scores
    metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
               'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
    
    scores = ["%1.3f" % results[m] for m in metrics]
    scores_str = ["%s: %1.3f" % (m, results[m]) for m in metrics]
    scores_combined[ckpt_str] = results
    
    valid_ckpt_exists = valid_ppl_dict is None \
                        or checkpoint not in valid_ppl_dict.keys()
    test_ckpt_exists = test_ppl_dict is None \
                        or checkpoint not in test_ppl_dict.keys()
    
    score_file = os.path.join(infer_config.eval_log_path, 'metric_scores')
    
    # Finally write scores to file
    with open(score_file + ".txt", 'a') as f:
        out_string = "===================================\r\n"
        out_string += "%s\r\n" % ckpt_file
        out_string += "===================================\r\n"
        out_string += "%s\r\n" % "\r\n".join(scores_str)
        out_string += "Perplexity (valid): "
        
        if valid_ckpt_exists:
            out_string += "N/A\r\n"
        else:
            out_string += "%2.3f\r\n" % valid_ppl_dict[checkpoint]
            
        out_string += "Perplexity (test): "
        if test_ckpt_exists:
            out_string += "N/A\r\n"
        else:
            out_string += "%2.3f\r\n" % test_ppl_dict[checkpoint]
        out_string += "\r\n\r\n"
        f.write(out_string)
    
    # Write scores to file in CSV style
    with open(score_file + ".csv", 'a') as f:
        out_string = "%s," % ckpt_str
        out_string += "%s," % ",".join(scores)
        if valid_ckpt_exists:
            out_string += "N/A,"
        else:
            out_string += "%2.3f," % valid_ppl_dict[checkpoint]
        if test_ckpt_exists:
            out_string += "N/A\r\n"
        else:
            out_string += "%2.3f\r\n" % test_ppl_dict[checkpoint]
        f.write(out_string)
    
    # Write individual scores
    sorted_cider = sorted(results['evalImgs'],
                          key=lambda k: k['CIDEr'],
                          reverse=True)
    json_file = score_file + "_detailed.json"
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            cider_dict = json.load(f)
        cider_dict[checkpoint] = sorted_cider
    else:
        cider_dict = {checkpoint:sorted_cider}
    with open(json_file, 'w') as f:
        json.dump(cider_dict, f)
    
    return scores_combined

