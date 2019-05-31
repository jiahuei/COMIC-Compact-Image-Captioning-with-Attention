#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:44:23 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os, sys, time, traceback as tb
from model import CaptionModel
import inputs.manager_image_caption as inputs
import configuration as conf
import ops
value_summary = ops.add_value_summary


def train_fn(config):
    """Main training function. To be called by `try_to_train()`."""
    
    print('TensorFlow version: r{}'.format(tf.__version__))
    print('INFO: Logging to `{}`.'.format(config.log_path))
    
    # Setup input pipeline & Build model
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(config.rand_seed)
        if config.token_type == 'radix':
            inputs_man = inputs.InputManager_Radix(config)
        elif config.token_type == 'char':
            inputs_man = inputs.InputManager_Char(config)
        else:
            inputs_man = inputs.InputManager(config)
        c = inputs_man.config
        
        num_batches = int(c.split_sizes['train'] / c.batch_size_train)
        lr = c.lr_start
        n_steps_log = int(num_batches / c.num_logs_per_epoch)
        
        with tf.name_scope('train'):
            m_train = CaptionModel(
                                c,
                                mode='train',
                                batch_ops=inputs_man.batch_train, 
                                reuse=False,
                                name='train')
            m_train.dset_size = c.split_sizes['train']
        
        with tf.name_scope('valid'):
            m_valid = CaptionModel(
                                c,
                                mode='eval',
                                batch_ops=inputs_man.batch_eval,
                                reuse=True,
                                name='valid')
            m_valid.dset_size = c.split_sizes['valid']
        
        init_fn = tf.global_variables_initializer()
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Model')
        model_saver = tf.train.Saver(var_list=model_vars,
                                     max_to_keep=c.max_saves)
        saver = tf.train.Saver(max_to_keep=2)
    
    r = c.per_process_gpu_memory_fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=r)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
    summary_writer = tf.summary.FileWriter(c.log_path, g)
    
    with sess:
        # Restore model from checkpoint if provided
        sess.run(init_fn)
        lr = m_train.restore_model(sess, saver, lr)
        g.finalize()
        #ops.get_model_size(scope_or_list=m_train._get_trainable_vars(),
        ops.get_model_size(scope_or_list='Model/decoder/rnn_decoder',
                           log_path=c.log_path)
        start_step = sess.run(m_train.global_step)
        n_steps_log = int(n_steps_log / 5)
        
        print('INFO: Graph constructed. Training begins now.')
        start_epoch = time.time()
        
        for step in xrange(start_step, c.max_step):
            epoch = int(step / num_batches) + 1
            
            # Write summary to disk once every `n_steps_log` steps
            if (step +1) % (n_steps_log * 5) == 0:
                ppl, summary, global_step, lr = sess.run(
                                                    [m_train.dec_log_ppl,
                                                     m_train.summary_op,
                                                     m_train.global_step,
                                                     m_train.lr])
                t = time.time() - start_epoch
                speed = (step + 1 - start_step) * c.batch_size_train / t
                print('   Training speed: {:7.2f} examples/sec.'.format(speed))
                summary_writer.add_summary(summary, global_step)
                value_summary({'train/speed' : speed},
                              summary_writer, global_step)
            # Quick logging
            elif (step +1) % n_steps_log == 0:
                ppl, global_step, lr = sess.run(
                                            [m_train.dec_log_ppl,
                                             m_train.global_step,
                                             m_train.lr])
                ppl = np.exp(ppl)
                logstr = 'Epoch {:2d} ~~ {:6.2f} %  ~  '.format(
                        epoch, ((step % num_batches) + 1) / num_batches * 100)
                logstr += 'Perplexity {:8.4f} ~ LR {:5.3e} ~ '.format(ppl, lr)
                logstr += 'Step {}'.format(global_step)
                print('   ' + logstr)
            else:
                ppl, global_step = sess.run([m_train.dec_log_ppl,
                                             m_train.global_step])
            
            if num_batches > 5000:
                save = (step +1) % int(num_batches / 2) == 0
            else:
                save = (step +1) % num_batches == 0
            save = save and (step + 100) < c.max_step
            
            # Evaluation and save model
            if save or (step +1) == c.max_step:
                model_saver.save(sess, c.save_path + '_compact', global_step)
                saver.save(sess, c.save_path, global_step)
                _run_eval_loop(sess, c, m_valid, summary_writer, global_step)
            
            if (step +1) % num_batches == 0:
                if c.legacy:
                    lr = _lr_reduce_check(config, epoch, lr)
                    m_train.update_lr(sess, lr)
                    sess.run(m_train.lr)
                t = time.time() - start_epoch
                print('\n\n>>> Epoch {:3d} complete'.format(epoch))
                print('>>> Time taken: {:10.2f} minutes\n\n'.format(t / 60))
                start_epoch = time.time()
                start_step = step + 1
        
        sess.close()
        print('\n\nINFO: Training completed.')


def _lr_reduce_check(config, epoch, learning_rate):
    """ Helper to reduce learning rate every n epochs."""
    if (learning_rate > config.lr_end 
        and epoch % config.reduce_lr_every_n_epochs == 0):
        learning_rate /= 2
        if learning_rate < config.lr_end:
            learning_rate = config.lr_end
    return learning_rate


def _run_eval_loop(session, c, m, summary_writer, global_step):
    """
    Wrapper for running the validation loop.
    Returns the average perplexity per word.
    """
    name = m.name
    assert m.dset_size % c.batch_size_eval == 0
    num_batches = int(m.dset_size / c.batch_size_eval)
    ppl_list = []
    print('\nEvaluating model...\n')
    
    for step in tqdm(range(num_batches), desc='evaluation', ncols=100):
        ppl = session.run(m.dec_log_ppl)
        ppl_list.append(ppl)
    avg_ppl = np.exp(np.mean(ppl_list))
    print('>>> {} perplexity per word: {:8.4f}\n'.format(name, avg_ppl))
    value_summary({'{}/perplexity'.format(name) : avg_ppl},
                  summary_writer, global_step)
    return avg_ppl


def try_to_train(train_fn, try_block=True, overwrite=False, **kargs):
    """Wrapper for the main training function."""
    config = conf.Config(**kargs)
    config.overwrite_safety_check(overwrite)
    #if config.resume_training:
    #    print('INFO: Resuming training from checkpoint.')
    #    fp = os.path.join(config.log_path, 'config.pkl')
    #    config = conf.load_config(fp)
    #    config.resume_training = True
    #    config.checkpoint_path = kargs.pop('log_path')
    #    config.lr_end = kargs.pop('lr_end')
    #    config.max_epoch = kargs.pop('max_epoch')
    #else:
    #    config.save_config_to_file()
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
            err_msg = 'Error occured:\r\n\r\n%s\r\n' % str(error_log[0])
            err_msg += '%s\r\n%s\r\n\r\n' % (str(error_log[1]), str(error_log[2]))
            err_msg += '\r\n\r\nTraceback stack:\r\n\r\n'
            for entry in traceback_extract:        
                err_msg += '%s\r\n' % str(entry)
            name = 'error__' + os.path.split(config.log_path)[1] + '.txt'
            with open(os.path.join(os.path.dirname(config.log_path), name), 'w') as f:
                f.write(err_msg)
            print('\nWARNING: An error has occurred.\n')
            print(err_msg)
            #tf.reset_default_graph()
    else:
        train_fn(config)


