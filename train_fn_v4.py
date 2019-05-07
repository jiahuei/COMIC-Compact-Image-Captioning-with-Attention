#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:44:23 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys, math, time, psutil, traceback as tb
import model_v4_0 as model
from utility_functions.captions import input_caption_baseN_v2 as input_man
from utility_functions.captions.configuration import Config
from utility_functions import ops_v3 as my_ops


add_value_summary = my_ops.add_value_summary


def train_fn(config):
    """Main training function. To be called by `try_to_train()`."""
    
    print("TensorFlow version: r{}".format(tf.__version__))
    print("INFO: Logging to `%s`." % config.log_path)
    tf.set_random_seed(config.rand_seed)
    inputs_man = input_man.InputManager(config)
    config = inputs_man.config
    
    # Setup input pipeline & Build model
    g = tf.Graph()
    with g.as_default():
        inputs_man.initialise_tf_queue()
        
        with tf.name_scope("train"):
            model_train = model.Model(
                                config,
                                mode='train',
                                batch_ops=inputs_man.train_batch_ops, 
                                reuse=False,
                                name='train')
            model_train.dset_size = inputs_man.dset_sizes[0]
        
        with tf.name_scope("valid"):
            model_valid = model.Model(
                                config,
                                mode='eval',
                                batch_ops=inputs_man.valid_batch_ops,
                                reuse=True,
                                name='validation')
            model_valid.dset_size = inputs_man.dset_sizes[1]
        
        with tf.name_scope("test"):
            model_test = model.Model(
                                config,
                                mode='eval',
                                batch_ops=inputs_man.test_batch_ops,
                                reuse=True,
                                name='test')
            model_test.dset_size = inputs_man.dset_sizes[2]
        
        init_fn = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=None)
        coord = tf.train.Coordinator()
    
    r = config.per_process_gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
    summary_writer = tf.summary.FileWriter(config.log_path, g)
    
    num_batches = int(math.ceil(config.train_size / config.batch_size))
    max_step = num_batches * config.max_epoch
    lr = config.lr_start
    n_steps_log = int(num_batches / config.num_logs_per_epoch)
    n_steps_save = int(num_batches / config.num_saves_per_epoch)
    saves = 0
    
    with sess:
        # Restore model from checkpoint if provided
        sess.run(init_fn)
        runners = tf.train.start_queue_runners(coord=coord, sess=sess)
        lr = model_train.restore_model(sess, saver, lr)
        g.finalize()
        my_ops.get_model_size(scope_or_list='Model/decoder', log_path=config.log_path)
        start_step = sess.run(model_train.global_step)
        
        print("INFO: Graph constructed. Training begins now.")
        start_epoch = time.time()
        
        for step in xrange(start_step, max_step):
            _, ppl, logits, summary, global_step = sess.run(
                                                    [model_train.optimise,
                                                     model_train.loss,
                                                     model_train.inference,
                                                     model_train.summary_op,
                                                     model_train.global_step])
            ppl = np.exp(ppl)
            epoch = int(step / num_batches) +1
            logstr = ("Epoch %2d - %3.2f %%  -  "
                      "Perplexity %3.4f - LR %1.1e" % 
                        (epoch, ((step % num_batches)+1) / num_batches*100, 
                         ppl, lr))
            
            # Quick logging
            if (step +1) % int(n_steps_log / 5) == 0:
                print("   " + logstr)
#            if (step +1) % int(n_steps_log * 5) == 0:
#                logstr += "\r\n\r\n%s" % str(psutil.virtual_memory())
#                _log_to_dropbox(config, logits, logstr)
            
            # Write summary to disk once every `n_steps_log` steps
            if (step +1) % n_steps_log == 0:
                t = time.time() - start_epoch
                speed = (step + 1 - start_step) * config.batch_size / t
                print("   Training speed: %4.2f examples/sec." % speed)
                summary_writer.add_summary(summary, global_step)
                add_value_summary({"train/speed" : speed},
                                  summary_writer,
                                  global_step)
            
            # Save model & run evaluation once every `n_steps_save` steps
            if ((step +1) % n_steps_save == 0
                    and (saves +1) % config.num_saves_per_epoch != 0):
                saves += 1
                saver.save(sess, config.save_path, global_step)
                
                ppl_valid = _run_eval_loop(sess, model_valid, config)
                ppl_test = _run_eval_loop(sess, model_test, config)
                add_value_summary({"loss/perplexity_valid" : ppl_valid,
                                    "loss/perplexity_test" : ppl_test},
                                  summary_writer,
                                  global_step)
            
            # Run once per epoch
            if (step +1) % num_batches == 0 or (step +1) == max_step:
                saves = 0
                saver.save(sess, config.save_path, global_step)
                t = time.time() - start_epoch
                print("\n\n>>> Epoch %3d complete" % epoch)
                print("Time taken: %10d seconds\n\n" % t)
                
                ppl_valid = _run_eval_loop(sess, model_valid, config)
                ppl_test = _run_eval_loop(sess, model_test, config)
                add_value_summary({"loss/perplexity_valid" : ppl_valid,
                                    "loss/perplexity_test" : ppl_test},
                                  summary_writer,
                                  global_step)
                
                lr = _lr_reduce_check(config, epoch, lr)
                model_train.update_lr(sess, lr)
                sess.run(model_train.lr)
                start_epoch = time.time()
                start_step = step
            
        # Shutdown everything to avoid zombies
        coord.request_stop()
        coord.join(runners)
        sess.close()
        print("\n\nINFO: Training completed.")


def _lr_reduce_check(config, epoch, learning_rate):
    """ Helper to reduce learning rate every n epochs."""
    if (learning_rate > config.lr_end 
        and epoch % config.reduce_lr_every_n_epochs == 0):
        learning_rate /= 2
        if learning_rate < config.lr_end:
            learning_rate = config.lr_end
    return learning_rate


def _run_eval_loop(session, model, config):
    """
    Wrapper for running the validation loop.
    Returns the average perplexity per word.
    """
    num_examples = model.dset_size
    name = model.name
    num_batches = int(math.ceil(num_examples / config.batch_size))
    ppl_list = []
    print("\n")
    for step in xrange(num_batches):
        if step % int(num_batches / 5) == 0:
            print("\tRunning %s step %5d" % (name, step))
        ppl = session.run(model.loss)
        ppl_list.append(ppl)
    avg_ppl_per_word = np.exp(np.mean(ppl_list))
    print(">>> %s perplexity per word: %3.4f" % (name, avg_ppl_per_word))
    return avg_ppl_per_word


def _log_to_dropbox(config, logits, logstr):
    """Helper to log some info to Dropbox."""
    logits = np.argmax(logits, 2)
    logstr = "\r\n" + logstr
    logstr += "\r\n\r\n\r\nExample sentences:\r\n\r\n"
    for i in range(10):
        sent = ' '.join([config.itow[str(id)] for id in logits[i, :]])
        logstr += "%s\r\n" % sent
    with open(config.dropbox_log_path + '/step.txt', 'w') as f:
        f.write(logstr)


def try_to_train(try_block=True, overwrite=False, **kargs):
    """Wrapper for the main training function."""
    config = Config(**kargs)
    config.overwrite_safety_check(overwrite)
    if config.resume_training:
        print("INFO: Resuming training from checkpoint.")
        fp = os.path.join(config.log_path, 'config.pkl')
        config = input_man.load_config(fp)
        config.lr_start = None
        config.resume_training = True
        config.checkpoint_path = kargs.pop('data_paths')['log_path']
        config.lr_end = kargs.pop('lr_end')
        config.max_epoch = kargs.pop('max_epoch')
    else:
        config.save_config_to_file()
    if try_block:
        try:
            train_fn(config)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            error_log = sys.exc_info()
            traceback_extract = tb.format_list(tb.extract_tb(error_log[2]))
            if not os.path.exists(config.log_path):
                os.makedirs(config.log_path)
            err_msg = "Error occured:\r\n\r\n%s\r\n" % str(error_log[0])
            err_msg += "%s\r\n%s\r\n\r\n" % (str(error_log[1]), str(error_log[2]))
            err_msg += "\r\n\r\nTraceback stack:\r\n\r\n"
            for entry in traceback_extract:        
                err_msg += "%s\r\n" % str(entry)
            with open(config.log_path + '/error_log.txt', 'w') as f:
                f.write(err_msg)
            print("\nWARNING: An error has occurred.\n")
            print(err_msg)
            #tf.reset_default_graph()
    else:
        train_fn(config)


