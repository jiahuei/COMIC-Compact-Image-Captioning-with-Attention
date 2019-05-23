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
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops, nn_ops, array_ops#, rnn_cell_impl
#from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
#from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
#from tensorflow.python.layers.core import Dense
from packaging import version
slim = tf.contrib.slim


_DEBUG = False


def dprint(string, is_debug):
    if is_debug: print('\n-- DEBUG: {}'.format(string))


def _dprint(string):
    return dprint(string, _DEBUG)


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


def get_linear_var(scope,
                   weight_shape,
                   bias_init=0.0):
    """
    Helper to create linear projection variables.
    
    Args:
        scope: name or scope.
        weight_shape: A 1-D tensor / list of ints specifying shape of 
            the weight variable.
        bias_init: Initial value of the bias variable. Pass in `None` if bias
            variable is not needed.
    
    Returns:
        Tensors (weight, bias) or (weight,).
    """
    with tf.variable_scope(scope):
        weight = tf.get_variable(name="weight",
                                 shape=weight_shape,
                                 dtype=tf.float32,
                                 initializer=None,
                                 trainable=True)
        if bias_init is None:
            return weight,
        else:
            bias = tf.get_variable(
                                name="bias",
                                shape=weight_shape[1],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_init),
                                trainable=True)
            return weight, bias


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
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        outputs = tf.contrib.layers.layer_norm(
                                            inputs=inputs,
                                            begin_norm_axis=begin_norm_axis,
                                            begin_params_axis=-1,
                                            **ln_kwargs)
    else:
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


#def ln_act_fn(scope, activation_fn):
#    return lambda x: layer_norm_activate(scope, x, activation_fn)


def c2d(scope,
        inputs,
        filter_shape,
        strides,
        padding='SAME',
        bias_init=0.0,
        data_format='NCHW',
        activation_fn=None):
    """
    Helper to perform 2-D convolution.
    
    Args:
        scope: name or scope.
        inputs: A 4-D tensor.
        filter_shape: Shape of the 2D convolution kernel:
            [filter_height, filter_width, in_channels, out_channels].
        strides: A list of ints. 1-D of length 4.
            The stride of the sliding window for each dimension of input.
        padding: A string, either "VALID" or "SAME".
        bias_init: Initial value of biases, float32. If None, biases are not
            added to the convolution output. Defaults to 0.
        data_format: A string from: "NHWC", "NCHW". Defaults to "NCHW".
        activation_fn: Activation function to use. Defaults to None.
    
    Returns:
        A Tensor. Has the same type as input. A 4-D tensor.
        The dimension order is determined by the value of data_format.
    """
    
    with tf.variable_scope(scope):
        weights = tf.get_variable(name="weights",
                                  shape=filter_shape,
                                  dtype=tf.float32,
                                  initializer=None,
                                  trainable=True)
        
        outputs = tf.nn.conv2d(input=inputs,
                               filter=weights,
                               strides=strides,
                               padding=padding,
                               data_format=data_format)
        
        if bias_init is not None:
            biases = tf.get_variable(
                            name="biases",
                            shape=[filter_shape[3]],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init),
                            trainable=True)
            outputs = tf.nn.bias_add(outputs, biases, data_format)
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
    return outputs


def separable_c2d(scope,
                  inputs,
                  filter_shape,
                  out_channels,
                  atrous_rate,
                  strides,
                  padding='SAME',
                  bias_init=0.0,
                  data_format='NCHW',
                  activation_fn=None):
    """
    Helper to perform 2-D convolution with separable filters.
    
    Performs a depthwise convolution that acts separately on channels
    followed by a pointwise convolution that mixes channels.
    
    Args:
        scope: name or scope.
        inputs: A 4-D tensor.
        filter_shape: Shape of the depthwise 2D convolution kernel:
            [filter_height, filter_width, in_channels, channel_multiplier].
        out_channels: Size of output channel dimension.
        atrous_rate: 1-D of size 2. The dilation rate in which we sample
            input values across the height and width dimensions in atrous
            convolution. If it is greater than 1, then all values of strides
            must be 1.
        strides: A list of ints. 1-D of length 4.
            The stride of the sliding window for each dimension of input.
            Applies to depthwise convolution only.
        padding: A string, either "VALID" or "SAME".
        bias_init: Initial value of biases, float32. If None, biases are not
            added to the convolution output. Defaults to 0.
        data_format: A string from: "NHWC", "NCHW". Defaults to "NCHW".
        activation_fn: Activation function to use. Defaults to None.
    
    Returns:
        A Tensor. Has the same type as input. A 4-D tensor.
        The dimension order is determined by the value of data_format.
    """
    
    with tf.variable_scope(scope):
        dw_weights = tf.get_variable(name="depthwise_weights",
                                     shape=filter_shape,
                                     dtype=tf.float32,
                                     initializer=None,
                                     trainable=True)
        pw_shape = [1, 1, filter_shape[2] * filter_shape[3], out_channels]
        pw_weights = tf.get_variable(name="pointwise_weights",
                                     shape=pw_shape,
                                     dtype=tf.float32,
                                     initializer=None,
                                     trainable=True)
        
        outputs = tf.nn.separable_conv2d(
                                input=inputs,
                                depthwise_filter=dw_weights,
                                pointwise_filter=pw_weights,
                                strides=strides,
                                padding=padding,
                                rate=atrous_rate,
                                data_format=data_format)
        
        if bias_init is not None:
            biases = tf.get_variable(
                            name="biases",
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init),
                            trainable=True)
            outputs = tf.nn.bias_add(outputs, biases, data_format)
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
    return outputs


def c2d_transpose(scope,
                  inputs,
                  filter_shape,
                  output_shape,
                  strides,
                  padding='SAME',
                  bias_init=0.0,
                  data_format='NCHW',
                  activation_fn=None):
    """
    Performs 2D transposed-convolution.
    
    For 1D transposed convolution:
        `inputs` is a 4-D tensor of shape [batch, in_channels, 1, width] or
            [batch, 1, width, channels].
        `strides` is a 1-D tensor with values [1, 1, 1, stride] or
            [1, 1, stride, 1].
    
    Args:
        scope: name or scope.
        inputs: A 4-D tensor.
        filter_shape: A 4-D Tensor with the same type as value 
            and shape [height, width, output_channels, in_channels].
            filter's `in_channels` dimension must match that of value.
        output_shape: A 1-D Tensor representing the output shape of the
            deconvolution op.
        strides: A list of ints. 1-D of length 4.
            The stride of the sliding window for each dimension of input.
        padding: A string, either "VALID" or "SAME".
        bias_init: Initial value of biases, float32. If None, biases are not
            added to the convolution output. Defaults to 0.
        data_format: A string from: "NHWC", "NCHW". Defaults to "NCHW".
        activation_fn: Activation function to use. Defaults to None.
    
    Returns:
        A tensor of shape specified by `output_shape`.
    """
    
    with tf.variable_scope(scope):
        weights = tf.get_variable(name="weights",
                                  shape=filter_shape,
                                  dtype=tf.float32,
                                  initializer=None,
                                  trainable=True)
        
        outputs = tf.nn.conv2d_transpose(
                                    value=inputs,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format)
        
        if bias_init is not None:
            biases = tf.get_variable(
                            name="biases",
                            shape=[filter_shape[2]],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init),
                            trainable=True)
            outputs = tf.nn.bias_add(outputs, biases, data_format)
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
    return outputs


def atrous_c2d_transpose(scope,
                         inputs,
                         filter_shape,
                         output_shape,
                         rate,
                         padding='SAME',
                         bias_init=0.0,
                         data_format='NCHW',
                         activation_fn=None):
    """
    Performs atrous 2D transposed-convolution.
    
    Atrous convolution is equivalent to standard convolution with
    upsampled filters with effective height
    `filter_height + (filter_height - 1) * (rate - 1)` and
    effective width `filter_width + (filter_width - 1) * (rate - 1)`,
    produced by inserting `rate - 1` zeros along consecutive elements
    across the `filters`' spatial dimensions.
    
    For 1D transposed convolution:
        `inputs` is a 4-D tensor of shape [batch, in_channels, 1, width] or
            [batch, 1, width, channels].
    
    NOTE: Since the underlying `atrous_conv2d_transpose` function only accepts
    "NHWC" format, we perform two transpose operations before and after the
    convolutions. Thus "NCHW" format is less efficient.
    
    Args:
        scope: name or scope.
        inputs: A 4-D tensor.
        filter_shape: A 4-D Tensor with the same type as value 
            and shape [height, width, output_channels, in_channels].
            filter's `in_channels` dimension must match that of value.
        output_shape: A 1-D Tensor representing the output shape of the
            deconvolution op.
        rate: A positive int32. The stride with which we sample
            input values across the `height` and `width` dimensions.
            Equivalently, the rate by which we upsample the filter values
            by inserting zeros across the `height` and `width` dimensions.
            In the literature, the same parameter is sometimes called
            `input stride` or `dilation`.
        padding: A string, either "VALID" or "SAME".
        bias_init: Initial value of biases, float32. If None, biases are not
            added to the convolution output. Defaults to 0.
        data_format: A string from: "NHWC", "NCHW". Defaults to "NCHW".
        activation_fn: Activation function to use. Defaults to None.
    
    Returns:
        A tensor of shape specified by `output_shape`.
    """
    
    with tf.variable_scope(scope):
        # atrous_conv2d_transpose only accepts "NHWC" format
        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
        
        weights = tf.get_variable(name="weights",
                                  shape=filter_shape,
                                  dtype=tf.float32,
                                  initializer=None,
                                  trainable=True)
        
        outputs = tf.nn.atrous_conv2d_transpose(
                                    value=inputs,
                                    filters=weights,
                                    output_shape=output_shape,
                                    rate=rate,
                                    padding=padding)
        
        if bias_init is not None:
            biases = tf.get_variable(
                            name="biases",
                            shape=[filter_shape[2]],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init),
                            trainable=True)
            outputs = tf.nn.bias_add(outputs, biases, 'NHWC')
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
        if data_format == 'NCHW':
            outputs = tf.transpose(outputs, perm=[0, 3, 1, 2])
    return outputs


def depthwise_c2d_transpose(scope,
                            inputs,
                            filter_shape,
                            output_shape,
                            strides,
                            padding='SAME',
                            bias_init=0.0,
                            data_format='NCHW',
                            activation_fn=None):
    """
    Performs 2D depthwise transposed-convolution.
    
    This operation computes the gradients of depthwise convolution.
    
    Given an input tensor of shape [batch, height, width, channels]
    and a filter tensor of shape [filter_height, filter_width, in_channels,
    channel_multiplier] containing in_channels convolutional filters of
    depth 1, depthwise_conv2d applies a different filter to each
    input channel (expanding from 1 channel to channel_multiplier channels
    for each), then concatenates the results together. The output has
    in_channels * channel_multiplier channels.
    
    For 1D transposed convolution:
        `inputs` is a 4-D tensor of shape [batch, channels, 1, width] or
            [batch, 1, width, channels].
        `strides` is a 1-D tensor with values [1, 1, 1, stride] or
            [1, 1, stride, 1].
    
    Args:
        scope: name or scope.
        inputs: A 4-D tensor.
        filter_shape: A 4-D Tensor with the same type as value and shape
            [filter_height, filter_width, in_channels, depthwise_multiplier].
            filter's `in_channels` dimension must match that of value.
        output_shape: A 1-D Tensor representing the output shape of the
            deconvolution op.
        strides: A list of ints. 1-D of length 4.
            The stride of the sliding window for each dimension of input.
        padding: A string, either "VALID" or "SAME".
        bias_init: Initial value of biases, float32. If None, biases are not
            added to the convolution output. Defaults to 0.
        data_format: A string from: "NHWC", "NCHW". Defaults to "NCHW".
        activation_fn: Activation function to use. Defaults to None.
    
    Returns:
        A tensor of shape specified by `output_shape`.
    """
    
    with tf.variable_scope(scope):
        weights = tf.get_variable(name="weights",
                                  shape=filter_shape,
                                  dtype=tf.float32,
                                  initializer=None,
                                  trainable=True)
        output_shape = tf.convert_to_tensor(output_shape)
        
        # Gradients for depthwise convolution
        outputs = gen_nn_ops.depthwise_conv2d_native_backprop_input(
                                    input_sizes=output_shape,
                                    filter=weights,
                                    out_backprop=inputs,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format,
                                    name=None)
        
        if bias_init is not None:
            if data_format == 'NHWC':
                out_channels = output_shape[3]
            else:
                out_channels = output_shape[1]
            biases = tf.get_variable(
                            name="biases",
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init),
                            trainable=True)
            outputs = tf.nn.bias_add(outputs, biases, 'NHWC')
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
    
    return outputs


def separable_c2d_transpose(scope,
                            inputs,
                            filter_shape,
                            output_shape,
                            strides,
                            padding='SAME',
                            bias_init=0.0,
                            data_format='NCHW',
                            activation_fn=None):
    """
    Performs 2D depthwise transposed-convolution followed by a
    pointwise convolution that mixes channels.
    
    If the input `channels` dimension is not equal to output `channels`
    dimension, then it will be scaled by the pointwise convolution.
    
    For 1D transposed convolution:
        `inputs` is a 4-D tensor of shape [batch, channels, 1, width] or
            [batch, 1, width, channels].
        `strides` is a 1-D tensor with values [1, 1, 1, stride] or
            [1, 1, stride, 1].
    
    Args:
        scope: name or scope.
        inputs: A 4-D tensor.
        filter_shape: A 4-D Tensor with the same type as value and shape
            [filter_height, filter_width, in_channels, depthwise_multiplier].
            filter's `in_channels` dimension must match that of value.
        output_shape: A 1-D Tensor representing the output shape of the
            deconvolution op.
        strides: A list of ints. 1-D of length 4.
            The stride of the sliding window for each dimension of input.
        padding: A string, either "VALID" or "SAME".
        bias_init: Initial value of biases, float32. If None, biases are not
            added to the convolution output. Defaults to 0.
        data_format: A string from: "NHWC", "NCHW". Defaults to "NCHW".
        activation_fn: Activation function to use. Defaults to None.
    
    Returns:
        A tensor of shape specified by `output_shape`.
    """
    in_shape = shape(inputs)
    dw_outputs_shape = output_shape
    if data_format == 'NCHW':
        pw_shape = [1, 1, in_shape[1], output_shape[1]]
        dw_outputs_shape[1] = in_shape[1]
    else:
        pw_shape = [1, 1, in_shape[3], output_shape[3]]
        dw_outputs_shape[3] = in_shape[3]
    
    with tf.variable_scope(scope):
        dw_weights = tf.get_variable(name="dw_conv/weights",
                                    shape=filter_shape,
                                    dtype=tf.float32,
                                    initializer=None,
                                    trainable=True)
        output_shape = tf.convert_to_tensor(output_shape)
        
        # Gradients for depthwise convolution
        dw_outputs = gen_nn_ops.depthwise_conv2d_native_backprop_input(
                                    input_sizes=dw_outputs_shape,
                                    filter=dw_weights,
                                    out_backprop=inputs,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format,
                                    name=None)
        # Pointwise 1x1 convolution
        outputs = c2d(scope="pw_conv",
                      inputs=dw_outputs,
                      filter_shape=pw_shape,
                      strides=[1, 1, 1, 1],
                      padding='SAME',
                      bias_init=bias_init,
                      data_format=data_format,
                      activation_fn=activation_fn)
    
    return outputs


@ops.RegisterGradient("DepthwiseConv2dNativeBackpropInput")
def _DepthwiseConv2dNativeBackpropInputGrad(op, grad):
    """The derivatives for deconvolution.
    Args:
        op: the Deconvolution op.
        grad: the tensor representing the gradient w.r.t. the output
    Returns:
        the gradients w.r.t. the input and the filter
    """
    return [None,
            nn_ops.depthwise_conv2d_native_backprop_filter(
                                grad,
                                array_ops.shape(op.inputs[1]),
                                op.inputs[2],
                                op.get_attr("strides"),
                                op.get_attr("padding"),
                                op.get_attr("data_format")),
            nn_ops.depthwise_conv2d_native(
                                grad,
                                op.inputs[1],
                                op.get_attr("strides"),
                                op.get_attr("padding"),
                                op.get_attr("data_format"))]


# Aliases
sep_c2d = separable_c2d
c2d_trans = c2d_transpose
atr_c2d_trans = atrous_c2d_transpose
dw_c2d_trans = depthwise_c2d_transpose
sep_c2d_trans = separable_c2d_transpose


def c3d(scope,
        inputs,
        filter_shape,
        strides,
        padding='SAME',
        bias_init=0.0,
        data_format='NDHWC',
        activation_fn=None):
    """
    Helper to perform 3-D convolution.
    
    Args:
        scope: name or scope.
        inputs: A 5-D tensor.
        filter_shape: Shape of the 3D convolution kernel: [filter_depth,
            filter_height, filter_width, in_channels, out_channels].
        strides: A list of ints. 1-D of length 5. Must have
            strides[0] = strides[4] = 1. The stride of the sliding window
            for each dimension of input.
        padding: A string, either "VALID" or "SAME".
        bias_init: Initial value of biases, float32. If None, biases are not
            added to the convolution output. Defaults to 0.
        data_format: A string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
        activation_fn: Activation function to use. Defaults to None.
    
    Returns:
        A Tensor. Has the same type as input. A 5-D tensor.
        The dimension order is determined by the value of data_format.
    """
    
    with tf.variable_scope(scope):
        weights = tf.get_variable(name="weights",
                                  shape=filter_shape,
                                  dtype=tf.float32,
                                  initializer=None,
                                  trainable=True)
        
        outputs = tf.nn.conv3d(input=inputs,
                               filter=weights,
                               strides=strides,
                               padding=padding,
                               data_format=data_format)
        
        if bias_init is not None:
            biases = tf.get_variable(
                            name="biases",
                            shape=[filter_shape[4]],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init),
                            trainable=True)
            if data_format == 'NDHWC':
                outputs = tf.nn.bias_add(outputs, biases, 'NHWC')
            else:
                biases = tf.reshape(biases, [1, -1, 1, 1, 1])
                outputs += biases
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
    return outputs


def depthwise_c3d(scope,
                  inputs,
                  filter_shape,
                  out_channels,
                  atrous_rate,
                  strides,
                  padding='SAME',
                  bias_init=0.0,
                  activation_fn=None):
    """
    Helper to perform depthwise 3-D convolution.
    
    Performs a depthwise convolution that acts separately on channels
    followed by a pointwise convolution that mixes channels.
    
    NOTE: `data_format` must be 'NDHWC'.
    
    Args:
        scope: name or scope.
        inputs: A 5-D tensor.
        filter_shape: Shape of the 3D convolution kernel: [filter_depth,
            filter_height, filter_width, in_channels, channel_multiplier].
        out_channels: Size of output channel dimension.
        atrous_rate: List / 1-D of size 3. The dilation rate in which we
            sample input values across [depth, height, width] dimensions in
            atrous convolution. If it is greater than 1, then all values of
            strides must be 1.
        strides: List / 1-D of length 3. Strides for [depth, height, width].
            Applies to depthwise convolution only.
        padding: A string, either "VALID" or "SAME".
        bias_init: Initial value of biases, float32. If None, biases are not
            added to the convolution output. Defaults to 0.
        activation_fn: Activation function to use. Defaults to None.
    
    Returns:
        A Tensor. Has the same type as input. A 5-D tensor.
        The dimension order is determined by the value of data_format.
    """
    
    assert len(strides) == 3
    assert len(atrous_rate) == 3
    #atrous_rate = [1] + atrous_rate + [1]
    #strides = [1] + strides + [1]
    
    with tf.variable_scope(scope):
        weights = tf.get_variable(
                            name="depthwise_height_weights",
                            shape=filter_shape,
                            dtype=tf.float32,
                            initializer=None,
                            trainable=True)
        pw_shape = [1, 1, 1, filter_shape[3] * filter_shape[4], out_channels]
        pw_weights = tf.get_variable(name="pointwise_channel_weights",
                                     shape=pw_shape,
                                     dtype=tf.float32,
                                     initializer=None,
                                     trainable=True)
        
        outputs = []
        for idx in range(filter_shape[3]):
            weight = tf.slice(weights, [0, 0, 0, idx, 0], [-1, -1, -1, 1, -1])
            in_slice = tf.slice(inputs, [0, 0, 0, 0, idx], [-1, -1, -1, -1, 1])
            
            out_slice = tf.nn.convolution(
                            input=in_slice,
                            filter=weight,
                            padding=padding,
                            strides=strides,
                            dilation_rate=atrous_rate,
                            data_format='NDHWC')
            outputs.append(out_slice)
        
        # Concatenate
        outputs = tf.concat(outputs, axis=4)
        
        # Pointwise Channel-Mixing Convolution
        outputs = tf.nn.convolution(
                            input=outputs,
                            filter=pw_weights,
                            padding='SAME',
                            strides=[1, 1, 1],
                            dilation_rate=None,
                            data_format='NDHWC')
        
        if bias_init is not None:
            # With 'NHWC', bias tensor will be added to the last dimension.
            biases = tf.get_variable(
                            name="biases",
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init),
                            trainable=True)
            outputs = tf.nn.bias_add(outputs, biases, 'NHWC')
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
        return outputs


def separable_c3d(scope,
                  inputs,
                  filter_shape,
                  out_channels,
                  atrous_rate,
                  strides,
                  padding='SAME',
                  bias_init=0.0,
                  data_format='NDHWC',
                  activation_fn=None,
                  separation='TC'):
    """
    Helper to perform 3-D convolution with separable filters.
    
    Performs a depthwise convolution that acts separately on channels
    followed by a pointwise convolution that mixes channels.
    
    NOTE: `data_format` of 'NCDHW' is less eficient, as it involves two
    transpose operations before and after the convolutions.
    
    Args:
        scope: name or scope.
        inputs: A 5-D tensor.
        filter_shape: Shape of the 3D convolution kernel: [filter_depth,
            filter_height, filter_width, in_channels, channel_multiplier].
        out_channels: Size of output channel dimension.
        atrous_rate: List / 1-D of size 3. The dilation rate in which we
            sample input values across [depth, height, width] dimensions in
            atrous convolution. If it is greater than 1, then all values of
            strides must be 1.
        strides: List / 1-D of length 3. Strides for [depth, height, width].
            Applies to depthwise convolution only.
        padding: A string, either "VALID" or "SAME".
        bias_init: Initial value of biases, float32. If None, biases are not
            added to the convolution output. Defaults to 0.
        data_format: A string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
        activation_fn: Activation function to use. Defaults to None.
        separation: A string from "C", "TC", "STC".
            `C`: Depthwise convolution with a [depth, height, width] kernel.
            `TC`: Depthwise convolution with a [1, height, width] kernel,
                followed by [depth, 1, 1] kernel.
            `STC`: Depthwise convolution with a [1, height, 1] kernel,
                followed by [1, width, 1] and [depth, 1, 1] kernels.
    
    Returns:
        A Tensor. Has the same type as input. A 5-D tensor.
        The dimension order is determined by the value of data_format.
    """
    
    if data_format == 'NDHWC':
        pass
    elif data_format == 'NCDHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 4, 1])                          # (N, D, H, W, C)
    else:
        raise ValueError("`data_format` only accepts `NDHWC` and `NCDHW`.")
    
    assert len(strides) == 3
    assert len(atrous_rate) == 3
    assert separation in ["C", "TC", "STC"]
    
    if separation == 'C':
        outputs = depthwise_c3d(scope,
                                inputs,
                                filter_shape,
                                out_channels,
                                atrous_rate,
                                strides,
                                padding,
                                bias_init,
                                activation_fn)
        return outputs
    
    with tf.variable_scope(scope):
        if separation == 'TC':
            sw_shape = filter_shape[1:4] + [1]                                  # (H, W, in_channels, 1)
            dw_spatial_weights = tf.get_variable(
                                        name="depthwise_spatial_weights",
                                        shape=sw_shape,
                                        dtype=tf.float32,
                                        initializer=None,
                                        trainable=True)
        else:
            hw_shape = [filter_shape[1], 1] + [filter_shape[3], 1]              # (H, 1, in_channels, 1)
            ww_shape = [1] + filter_shape[2:4] + [1]                            # (1, W, in_channels, 1)
            dw_height_weights = tf.get_variable(
                                        name="depthwise_height_weights",
                                        shape=hw_shape,
                                        dtype=tf.float32,
                                        initializer=None,
                                        trainable=True)
            dw_width_weights = tf.get_variable(
                                        name="depthwise_width_weights",
                                        shape=ww_shape,
                                        dtype=tf.float32,
                                        initializer=None,
                                        trainable=True)
        tw_shape = [filter_shape[0], 1] + filter_shape[3:]                      # (D, 1, in_channels, channel_multiplier)
        dw_temporal_weights = tf.get_variable(
                                    name="depthwise_temporal_weights",
                                    shape=tw_shape,
                                    dtype=tf.float32,
                                    initializer=None,
                                    trainable=True)
        pw_shape = [1, 1, filter_shape[3] * filter_shape[4], out_channels]
        pw_weights = tf.get_variable(name="pointwise_channel_weights",
                                     shape=pw_shape,
                                     dtype=tf.float32,
                                     initializer=None,
                                     trainable=True)
        
        # Spatial Depthwise Convolution
        in_shape = shape(inputs)
        inputs = tf.reshape(inputs, [-1] + in_shape[2:])
        if separation == 'STC':
            outputs = tf.nn.depthwise_conv2d(
                                    input=inputs,
                                    filter=dw_height_weights,
                                    strides=[1, strides[1], 1, 1],
                                    padding=padding,
                                    rate=atrous_rate[1:],
                                    data_format='NHWC')
            outputs = tf.nn.depthwise_conv2d(
                                    input=outputs,
                                    filter=dw_width_weights,
                                    strides=[1, 1, strides[2], 1],
                                    padding=padding,
                                    rate=atrous_rate[1:],
                                    data_format='NHWC')
        else:
            outputs = tf.nn.depthwise_conv2d(
                                    input=inputs,
                                    filter=dw_spatial_weights,
                                    strides=[1, strides[1], strides[2], 1],
                                    padding=padding,
                                    rate=atrous_rate[1:],
                                    data_format='NHWC')
        
        # Temporal Depthwise Convolution
        spatial_size = shape(outputs)[1:3]
        outputs = tf.reshape(outputs, in_shape[:2] + [-1, in_shape[4]])         # (N, D, H * W, C)
        
        if strides[0] > 1:
            # We move the Height & Width axes to Batch axis, because
            # depthwise_conv2d only supports equal striding values.
            outputs = tf.transpose(outputs, [0, 2, 1, 3])                       # (N, H * W, D, C)
            outputs = tf.reshape(outputs, [-1, in_shape[1], in_shape[4]])       # (N * H * W, D, C)
            outputs = tf.expand_dims(outputs, axis=2)                           # (N * H * W, D, 1, C)
        outputs = tf.nn.depthwise_conv2d(
                                    input=outputs,
                                    filter=dw_temporal_weights,
                                    strides=[1, strides[0], strides[0], 1],
                                    padding=padding,
                                    rate=[atrous_rate[0], 1],
                                    data_format='NHWC')
        temporal_len = shape(outputs)[1]
        if strides[0] > 1:
            outputs = tf.squeeze(outputs)                                       # (N * H * W, D, C)
            out_shape = [in_shape[0], -1, temporal_len, in_shape[4]]
            outputs = tf.reshape(outputs, out_shape)                            # (N, H * W, D, C)
            outputs = tf.transpose(outputs, [0, 2, 1, 3])                       # (N, D, H * W, C)
        
        # Pointwise Channel-Mixing Convolution
        outputs = tf.nn.conv2d(
                            input=outputs,
                            filter=pw_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            use_cudnn_on_gpu=True,
                            data_format='NHWC')
        out_shape = [in_shape[0], temporal_len] + spatial_size + [out_channels]
        outputs = tf.reshape(outputs, out_shape)
        
        if bias_init is not None:
            biases = tf.get_variable(
                            name="biases",
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_init),
                            trainable=True)
            outputs = tf.nn.bias_add(outputs, biases, 'NHWC')
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
        if data_format == 'NCDHW':
            outputs = tf.transpose(outputs, [0, 4, 1, 2, 3])
    
    return outputs


# Aliases
sep_c3d = separable_c3d


