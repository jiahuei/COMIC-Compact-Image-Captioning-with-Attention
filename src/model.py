# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 17:12:32 2017

@author: jiahuei

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import tensorflow as tf
from model_base import ModelBase
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CURR_DIR, '..', 'common'))
import ops
_shape = ops.shape


_DEBUG = False
def _dprint(string):
    return ops.dprint(string, _DEBUG)


class CaptionModel(ModelBase):
    
    def __init__(self,
                 config,
                 mode, 
                 batch_ops=None,
                 reuse=False,
                 name=None):
        assert mode in ['train', 'eval', 'infer']
        print('INFO: Building graph for: {}'.format(name))
        super(CaptionModel, self).__init__(config)
        self.mode = mode
        self.batch_ops = batch_ops
        self.reuse = reuse
        self.name = name
        self._batch_size = _shape(self.batch_ops[0])[0]
        
        # Start to build the model
        c = self._config
        is_inference = self.mode == 'infer'
        vs_kwargs = dict(reuse=tf.AUTO_REUSE,
                         initializer=self._get_initialiser())
        
        if self.is_training():
            self._create_gstep()
            if c.legacy:
                self._create_lr()
            else:
                self._create_cosine_lr(c.max_step)
        
        with tf.variable_scope('Model', **vs_kwargs):
            self._process_inputs()
            with tf.variable_scope('encoder'):
                self._encoder()
            with tf.variable_scope('decoder'):
                self._decoder_rnn()
        
        # We place the optimisation graph out of 'Model' scope
        self._train_caption_model()
        
        if is_inference:
            attention_maps = self.dec_attn_maps
            if attention_maps is None:
                self.infer_output = [self.dec_preds, tf.zeros([])]
            else:
                self.infer_output = [self.dec_preds, attention_maps]
            return None
        
        # Log softmax temperature value
        t = tf.get_collection('softmax_temperatures')
        if len(t) > 0: tf.summary.scalar('softmax_temperature', t[0])
        self.summary_op = tf.summary.merge_all()
        print('INFO: Model `{}` initialisation complete.'.format(mode))


class CaptionModel_SCST(ModelBase):
    
    def __init__(self,
                 config,
                 scst_mode,
                 reuse=False):
        assert scst_mode in ['train', 'sample']
        #assert config.token_type == 'word'
        
        print('INFO: Building graph for: {}'.format(scst_mode))
        super(CaptionModel_SCST, self).__init__(config)
        self.mode = scst_mode if scst_mode == 'train' else 'infer'
        c = self._config
        batch_size = c.batch_size_train
        if self.is_training():
            batch_size *= (c.scst_beam_size +0)
        im_size = c.cnn_input_size
        self.imgs = tf.placeholder(
                    dtype=tf.float32,
                    shape=[batch_size, im_size[0], im_size[1], 3])
        self.captions = tf.placeholder_with_default(
                    input=tf.zeros(shape=[batch_size, 1], dtype=tf.int32),
                    shape=[batch_size, None])
        self.rewards = tf.placeholder(dtype=tf.float32, shape=[batch_size])
        self.batch_ops = [self.imgs, self.captions]
        self.reuse = reuse
        self.name = scst_mode
        self._batch_size = _shape(self.batch_ops[0])[0]
        
        # Start to build the model
        vs_kwargs = dict(reuse=tf.AUTO_REUSE,
                         initializer=self._get_initialiser())
        
        if self.is_training():
            self._create_gstep()
            self._create_cosine_lr(c.max_step)
        
        with tf.variable_scope('Model', **vs_kwargs):
            self._process_inputs()
            with tf.variable_scope('encoder'):
                self._encoder()
            with tf.variable_scope('decoder'):
                if self.is_training():
                    self._decoder_rnn_scst()
                else:
                    with tf.name_scope('greedy'):
                        self._decoder_rnn_scst(1)
                        self.dec_preds_greedy = self.dec_preds
                    with tf.name_scope('beam'):
                        self._decoder_rnn_scst(c.scst_beam_size)
                        self.dec_preds_beam = self.dec_preds
                    #with tf.name_scope('sample'):
                    #    self._decoder_rnn_scst(0)
                    #    self.dec_preds_sample = self.dec_preds
        
        # Generated captions can be obtained by calling self.dec_preds
        
        # We place the optimisation graph out of 'Model' scope
        self.train_scst = self._train_caption_model(scst=True)
        
        
        # Log softmax temperature value
        t = tf.get_collection('softmax_temperatures')
        if len(t) > 0: tf.summary.scalar('softmax_temperature', t[0])
        self.summary_op = tf.summary.merge_all()
        print('INFO: Model `{}` initialisation complete.'.format(scst_mode))
    
    
    