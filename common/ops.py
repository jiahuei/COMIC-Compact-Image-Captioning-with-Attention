#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:22:42 2017

@author: jiahuei
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools, os
import numpy as np
import tensorflow as tf
#from tensorflow.python.framework import ops
#from tensorflow.python.ops import gen_nn_ops, nn_ops, array_ops, rnn_cell_impl
#from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
#from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
#from tensorflow.python.layers.core import Dense
from packaging import version
slim = tf.contrib.slim


def number_to_base(n, base):
    """Function to convert any base-10 integer to base-N."""
    if base < 2:
        raise ValueError('Base cannot be less than 2.')
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


def shape(tensor, replace_None=False):
    """
    Returns the shape of the Tensor as list.
    Can also replace unknown dimension value from `None` to `-1`.
    """
    s = tensor.get_shape().as_list()
    if replace_None:
        s = [-1  if i == None else i for i in s]
    return s


def add_value_summary(data_dict, summary_writer, global_step):
    """Helper to write value to summary."""
    for name, value in data_dict.iteritems():
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        summary_writer.add_summary(summary, global_step)


def get_model_size(scope_or_list=None, log_path=None):
    if isinstance(scope_or_list, list):
        var = scope_or_list
    else:
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                scope=scope_or_list)
    var_shape = []
    for v in var:
        try:
            var_shape.append(shape(v))
        except:
            mssg = 'INFO: Model size calculation: Encountered opaque variable: '
            print(mssg + v.op.name)
            var_shape.append([])
    model_size = sum([np.prod(v) for v in var_shape])
    if isinstance(scope_or_list, list) or scope_or_list is None:
        name = 'List provided'
    else:
        name = 'Scope `{}`'.format(scope_or_list)
    mssg = "INFO: {} contains {:,d} trainable parameters.".format(
            name, int(model_size))
    print('\n{}\n'.format(mssg))
    mssg = '\r\n{}\r\n\r\n'.format(mssg)
    for i, v in enumerate(var):
        mssg += '{}\r\n{}\r\n\r\n'.format(v.op.name, var_shape[i])
    mssg += '\r\n\r\n'
    if log_path is not None:
        with open(os.path.join(log_path, 'model_size.txt'), 'a') as f:
            f.write(mssg)
    return model_size


###############################################################################


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    # https://danijar.github.io/structuring-your-tensorflow-models
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def def_var_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    # https://danijar.github.io/structuring-your-tensorflow-models
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


@doublewrap
def def_name_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.name_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    name scope. The scope name defaults to the name of the wrapped
    function.
    # https://danijar.github.io/structuring-your-tensorflow-models
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.name_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def lazy_property(function):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    # https://danijar.github.io/structuring-your-tensorflow-models
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


###############################################################################



def l2_regulariser(tensor,
                   weight_decay_factor):
    """A `Tensor` -> `Tensor` function that applies L2 weight loss."""
    weight_decay = tf.multiply(tf.nn.l2_loss(tensor),
                               weight_decay_factor,
                               name="L2_weight_loss")
    return weight_decay


def relu(tensor,
         relu_leak_factor):
    """Helper to perform leaky / regular ReLU operation."""
    with tf.name_scope("Leaky_Relu"):
        return tf.maximum(tensor, tensor * relu_leak_factor)


def linear(scope,
           inputs,
           output_dim,
           bias_init=0.0,
           activation_fn=None):
    """
    Helper to perform linear map with optional activation.
    
    Args:
        scope: name or scope.
        inputs: A 2-D tensor.
        output_dim: The output dimension (second dimension).
        bias_init: Initial value of the bias variable. Pass in `None` for
            linear projection without bias.
        activation_fn: Activation function to be used. (Optional)
    
    Returns:
        A tensor of shape [inputs_dim[0], `output_dim`].
    """
    with tf.variable_scope(scope):
        input_dim = shape(inputs)[1]
        weight = tf.get_variable(name="weight",
                                 shape=[input_dim, output_dim],
                                 dtype=tf.float32,
                                 initializer=None,
                                 trainable=True)
        if bias_init is None:
            output = tf.matmul(inputs, weight)
        else:
            bias = tf.get_variable(
                            name="bias",
                            shape=output_dim,
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init),
                            trainable=True)
            output = tf.matmul(inputs, weight) + bias
        if activation_fn is not None: 
            output = activation_fn(output)
        return output


def layer_norm_activate(scope,
                        inputs,
                        activation_fn=None,
                        begin_norm_axis=1):
    """
    Performs Layer Normalization followed by `activation_fn`.
    
    Args:
        scope: name or scope.
        inputs: A N-D tensor. 
        activation_fn: Activation function to be used. (Optional)
    
    Returns:
        A tensor of the same shape as `inputs`.
    """
    ln_kwargs = dict(
                    center=True,
                    scale=True, 
                    activation_fn=activation_fn,
                    reuse=False,
                    trainable=True,
                    scope=scope)
    if version.parse(tf.__version__) >= version.parse('1.9'):
        outputs = tf.contrib.layers.layer_norm(
                                            inputs=inputs,
                                            begin_norm_axis=begin_norm_axis,
                                            begin_params_axis=-1,
                                            **ln_kwargs)
    else:
        print('WARNING: `layer_norm_activate`: `begin_norm_axis` is ignored.')
        outputs = tf.contrib.layers.layer_norm(
                                            inputs=inputs,
                                            **ln_kwargs)
    return outputs


def batch_norm_activate(scope,
                        inputs,
                        is_training,
                        activation_fn=None,
                        data_format='NHWC'):
    """
    Performs Batch Normalization followed by `activation_fn`.
    
    Args:
        scope: name or scope.
        inputs: A N-D tensor. 
        activation_fn: Activation function to be used. (Optional)
    
    Returns:
        A tensor of the same shape as `inputs`.
    """
    
    batch_norm_params = dict(
            epsilon = 1e-3,
            decay = 0.9997,
            trainable = True,
            activation_fn = None,
            fused = True,
            updates_collections=tf.GraphKeys.UPDATE_OPS,
            center = True,
            scale = True,
            )
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            outputs = slim.batch_norm(inputs=inputs,
                                      is_training=is_training,
                                      data_format=data_format)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
    return outputs


