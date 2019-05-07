#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:43:38 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utility_functions import ops_v3 as my_ops


class _BaseModel(object):
    """
    Base for model implementations.
    """
    
    def __init__(self, config):
        assert config.lang_model in ['word', 'baseN', 'char', 'bpe']
        if config.lang_model == 'baseN':
            self._softmax_size = config.base + 2
        elif config.lang_model == 'bpe':
            self._softmax_size = 66
        else:
            self._softmax_size = len(config.itow)
        self._config = config
    
    
    def _get_initialiser(self):
        """Helper to select initialiser."""
        if self._config.initialiser == 'xavier':
            print("INFO: Using Xavier initialiser.")
            init = tf.contrib.slim.xavier_initializer()
        else:
            print("INFO: Using TensorFlow default initialiser.")
            init = None
        return init
    
    
    def _get_rnn_cell(self, rnn_size):
        """Helper to select RNN cell(s)."""
        if self._config.rnn == 'LSTM':
            cells = tf.contrib.rnn.BasicLSTMCell(
                                        num_units=rnn_size,
                                        state_is_tuple=True,
                                        reuse=self.reuse)
        elif self._config.rnn == 'LN_LSTM':
            cells = tf.contrib.rnn.LayerNormBasicLSTMCell(
                                        num_units=rnn_size,
                                        reuse=self.reuse)
        elif self._config.rnn == 'GRU':
            cells = tf.contrib.rnn.GRUCell(
                                        num_units=rnn_size,
                                        reuse=self.reuse)
        else:
            raise ValueError("Only `LSTM`, `LN_LSTM` and `GRU` are accepted.")
        if self._config.num_layers > 1:
            raise ValueError("RNN layer > 1 not implemented.")
            #cells = tf.contrib.rnn.MultiRNNCell([cells] * self.config.num_layers)
        
        # Setup input and output dropouts
        input_keep = 1 - self._config.dropout_i
        output_keep = 1 - self._config.dropout_o
        if self.is_training() and (input_keep < 1 or output_keep < 1):
            print("INFO: Training using dropout.")
            cells = tf.contrib.rnn.DropoutWrapper(cells,
                                                  input_keep_prob=input_keep,
                                                  output_keep_prob=output_keep)
        return cells


    def _get_rnn_init(self, image_features, cell):
        """
        Helper to generate initial state of RNN cell.
        """
        if self._config.rnn == 'LSTM' or self._config.rnn == 'LN_LSTM':
            init_state_h = self._pre_act_linear("rnn_initial_state",
                                                image_features,
                                                cell.state_size[1],
                                                tf.nn.tanh)
            initial_state = tf.contrib.rnn.LSTMStateTuple(
                                    init_state_h * 0.0, init_state_h)
        elif self._config.rnn == 'GRU':
            initial_state = self._pre_act_linear("rnn_initial_state",
                                                 image_features,
                                                 cell.state_size,
                                                 tf.nn.tanh)
        return initial_state
    
    
    def _create_lr_gstep(self):
        """
        Helper to create learning rate and global step variables.
        """
        
        self.global_step = tf.get_variable(
                                tf.GraphKeys.GLOBAL_STEP,
                                shape=[],
                                dtype=tf.int32,
                                initializer=tf.zeros_initializer(),
                                trainable=False,
                                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                             tf.GraphKeys.GLOBAL_STEP])
        self.lr = tf.get_variable(
                                'learning_rate',
                                shape=[],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer(),
                                trainable=False,
                                collections=[tf.GraphKeys.GLOBAL_VARIABLES])
        
        '''
        if self.is_training() is False: return
        self.global_step = tf.Variable(
                                initial_value=0,
                                name="global_step",
                                trainable=False,
                                collections=[tf.GraphKeys.GLOBAL_STEP, 
                                             tf.GraphKeys.GLOBAL_VARIABLES])
        
        self.lr = tf.Variable(0.0, name="learning_rate", trainable=False)
        '''
        self._new_step = tf.placeholder(tf.int32, None, "new_global_step")
        self._new_lr = tf.placeholder(tf.float32, None, "new_lr")
        self._assign_step = tf.assign(self.global_step, self._new_step)
        self._assign_lr = tf.assign(self.lr, self._new_lr)
    
    
    def _add_vars_summary(self):
        if self._config.add_vars_summary:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
    
    
    def update_lr(self, session, lr_value):
        session.run(self._assign_lr, {self._new_lr: lr_value})
    
    
    def get_global_step(self, session):
        return session.run(self.global_step)
    
    
    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"
    
    
    def _regulariser(self, var):
        """A `Tensor` -> `Tensor` function that applies L2 weight loss."""
        return my_ops.regulariser(var, self._config.weight_decay)
    
    
    def _relu(self, tensor):
        """Leaky / regular ReLU activation function."""
        return my_ops.relu(tensor, self._config.relu_leak)
    
    
    def _norm_act(self, scope, tensor, act_fn):
        return my_ops.normalise_activate(scope, tensor, act_fn)
    
    
    def _linear(self,
                scope,
                inputs,
                output_dim,
                bias_init=0.0, 
                relu=False):
        """
        Helper to perform linear map with optional ReLU activation.
        
        Args:
            scope: name or scope.
            inputs: A 2-D tensor.
            output_dim: The output dimension (second dimension).
            relu: Optional ReLU activation.
            bias_init: Initial value of the bias variable. Pass in `None` for
                linear projection without bias.
    
        Returns:
            A tensor of shape [inputs_dim[0], `output_dim`].
        """
        if relu:
            act_fn = self._relu
        else:
            act_fn = None
        return my_ops.linear(scope=scope,
                             inputs=inputs,
                             output_dim=output_dim,
                             bias_init=bias_init,
                             activation_fn=act_fn)
    
    
    def _pre_act_linear(self,
                        scope,
                        inputs,
                        output_dim,
                        act_fn=None):
        """
        Helper to perform layer norm, activation, followed by linear map.
        
        Args:
            scope: name or scope.
            inputs: A 2-D tensor.
            output_dim: The output dimension (second dimension).
            act_fn: Activation function to be used. (Optional)
    
        Returns:
            A tensor of shape [inputs_dim[0], `output_dim`].
        """
        with tf.variable_scope(scope):
            inputs = my_ops.normalise_activate("pre_act_LN", inputs, act_fn)
            return my_ops.linear(scope="linear",
                                 inputs=inputs,
                                 output_dim=output_dim,
                                 bias_init=None,
                                 activation_fn=None)

