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
from tensorflow.python.ops import gen_nn_ops, nn_ops, array_ops, rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.layers.core import Dense


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
    var_shape = [v.get_shape().as_list() for v in var]
    model_size = sum([np.prod(v) for v in var_shape])
    if isinstance(scope_or_list, list) or scope_or_list is None:
        name = 'Entire model'
    else:
        name = 'Scope `{}`'.format(scope_or_list)
    mssg = "\nINFO: {} contains {:,d} trainable parameters.\n".format(
            name, int(model_size))
    print(mssg)
    if log_path is not None:
        with open(os.path.join(log_path, 'model_size.txt'), 'w') as f:
            f.write('{}\r\n\r\n'.format(mssg))
            for v in var:
                f.write('{}\r\n{}\r\n\r\n'.format(
                        v.op.name, v.get_shape().as_list()))
    return model_size


###############################################################################


def shape_list(tensor):
    return tensor.get_shape().as_list()


def regulariser(tensor,
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
        input_dim = shape_list(inputs)[1]
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


def normalise_activate(scope,
                       inputs,
                       activation_fn=None):
    """
    Performs Layer Normalization followed by `activation_fn`.
    
    Args:
        scope: name or scope.
        inputs: A N-D tensor. 
        activation_fn: Activation function to be used. (Optional)
    
    Returns:
        A tensor of the same shape as `inputs`.
    """
    outputs = tf.contrib.layers.layer_norm(inputs=inputs,
                                           center=True,
                                           scale=True, 
                                           activation_fn=activation_fn,
                                           reuse=False,
                                           trainable=True,
                                           scope=scope)
    return outputs


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
    in_shape = shape_list(inputs)
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
        in_shape = shape_list(inputs)
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
        spatial_size = shape_list(outputs)[1:3]
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
        temporal_len = shape_list(outputs)[1]
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


###############################################################################


def rnn_decoder_beam_search(cell,
                            embedding_fn,
                            output_layer,
                            batch_size,
                            beam_size,
                            length_penalty_weight,
                            maximum_iterations,
                            start_id,
                            end_id,
                            swap_memory=True):
    """
    Dynamic RNN loop function for inference. Performs beam search.
    Operates in time-major mode.
    
    Args:
        cell: An `RNNCell` instance (with or without attention).
        embeddings: A float32 tensor of shape [time, batch, word_size].
        output_layer: An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Layer to apply to the RNN output prior to
            storing the result or sampling. Pass `None` to disable it.
        batch_size: Int scalar. Size of batch.
        beam_size: `Int scalar. Size of beam for beam search.
        length_penalty_weight: Float weight to penalise length.
            Disabled with 0.0.
        maximum_iterations: Int scalar. Maximum number of decoding steps.
        start_id: `int32` scalar, the token that marks start of decoding.
        end_id: `int32` scalar, the token that marks end of decoding.
        swap_memory: Python bool, whether GPU-CPU memory swap is enabled.
            Argument passed to `tf.while_loop`.
    
    Returns:
        top_sequence, top_score, None
    """
    print("INFO: Building subgraph for Beam Search.")
    
    state_init = cell.zero_state(batch_size * beam_size, tf.float32)
    start_ids = tf.tile([start_id], multiples=[batch_size])
    
    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
    #decoder = BeamSearchDecoder_v2(
                                cell=cell,
                                embedding=embedding_fn,
                                start_tokens=start_ids,
                                end_token=end_id,
                                initial_state=state_init,
                                beam_width=beam_size,
                                output_layer=output_layer,
                                length_penalty_weight=length_penalty_weight)
    dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                                decoder=decoder,
                                output_time_major=True,
                                impute_finished=False,                          # Problematic when used with BeamSearch
                                maximum_iterations=maximum_iterations,
                                parallel_iterations=32,
                                swap_memory=swap_memory)
    
    # `dec_outputs` will be a `FinalBeamSearchDecoderOutput` object
    # `dec_states` will be a `BeamSearchDecoderState` object
    predicted_ids = dec_outputs.predicted_ids                                   # (time, batch_size, beam_size)
    scores = dec_outputs.beam_search_decoder_output.scores                      # (time, batch_size, beam_size)
    top_sequence = predicted_ids[:, :, 0]
    top_score = scores[:, :, 0]                                                 # log-softmax scores
    
    return top_sequence, top_score, None


def rnn_decoder_greedy_search(cell,
                              embedding_fn,
                              output_layer,
                              batch_size,
                              maximum_iterations,
                              start_id,
                              end_id,
                              swap_memory=True):
    """
    Dynamic RNN loop function for inference. Performs greedy search.
    Operates in time-major mode.
    
    Args:
        cell: An `RNNCell` instance (with or without attention).
        embedding_fn: A callable that takes a vector tensor of `ids`
            (argmax ids), or the `params` argument for `embedding_lookup`.
            The returned tensor will be passed to the decoder input.
        output_layer: An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Layer to apply to the RNN output prior to
            storing the result or sampling. Pass `None` to disable it.
        batch_size: Int scalar. Size of batch.
        maximum_iterations: Int scalar. Maximum number of decoding steps.
        start_id: `int32` scalar, the token that marks start of decoding.
        end_id: `int32` scalar, the token that marks end of decoding.
        swap_memory: Python bool, whether GPU-CPU memory swap is enabled.
            Argument passed to `tf.while_loop`.
    
    Returns:
        output_ids, rnn_outputs, decoder_state
    """
    print("INFO: Building subgraph for Greedy Search.")
    
    # Initialise `AttentionWrapperState` with provided RNN state
    state_init = cell.zero_state(batch_size, tf.float32)
    start_ids = tf.tile([start_id], multiples=[batch_size])
    
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            embedding=embedding_fn,
                            start_tokens=start_ids,
                            end_token=end_id)
    decoder = tf.contrib.seq2seq.BasicDecoder(
                                cell=cell,
                                helper=helper,
                                initial_state=state_init,
                                output_layer=output_layer)
    dec_outputs, dec_states, _ = tf.contrib.seq2seq.dynamic_decode(
                                decoder=decoder,
                                output_time_major=True,
                                impute_finished=False,
                                maximum_iterations=maximum_iterations,
                                parallel_iterations=32,
                                swap_memory=swap_memory)
    
    # `dec_outputs` will be a `BasicDecoderOutput` object
    # `dec_states` may be a `AttentionWrapperState` object
    rnn_out = dec_outputs.rnn_output
    output_ids = dec_outputs.sample_id
    
    return output_ids, rnn_out, dec_states


def rnn_decoder_training(cell,
                         embeddings,
                         output_layer,
                         batch_size,
                         sequence_length,
                         swap_memory=True):
    """
    Dynamic RNN loop function for training. Operates in time-major mode.
    The decoder will run until <EOS> token is encountered.
    
    Args:
        cell: An `RNNCell` instance (with or without attention).
        embeddings: A float32 tensor of shape [time, batch, word_size].
        output_layer: An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Layer to apply to the RNN output prior to
            storing the result or sampling. Pass `None` to disable it.
        batch_size: Int scalar. Size of batch.
        sequence_length: An int32 vector tensor. Length of sequence.
        swap_memory: Python bool, whether GPU-CPU memory swap is enabled.
            Argument passed to `tf.while_loop`.
    
    Returns:
        output_ids, rnn_outputs, decoder_state
    """
    print("INFO: Building dynamic decode subgraph.")
    
    # Initialise `AttentionWrapperState` with provided RNN state
    state_init = cell.zero_state(batch_size, tf.float32)
    helper = tf.contrib.seq2seq.TrainingHelper(
                            inputs=embeddings,
                            sequence_length=sequence_length,
                            time_major=True)
    decoder = tf.contrib.seq2seq.BasicDecoder(
                                cell=cell,
                                helper=helper,
                                initial_state=state_init,
                                output_layer=output_layer)
    dec_outputs, dec_states, _ = tf.contrib.seq2seq.dynamic_decode(
                                decoder=decoder,
                                output_time_major=True,
                                impute_finished=True,
                                maximum_iterations=None,
                                parallel_iterations=32,
                                swap_memory=swap_memory)
    
    # `dec_outputs` will be a `BasicDecoderOutput` object
    # `dec_states` may be a `AttentionWrapperState` object
    rnn_out = dec_outputs.rnn_output
    output_ids = dec_outputs.sample_id
    
    # Perform padding by copying elements from the last time step.
    # This is skipped in inference mode.
    pad_time = tf.shape(embeddings)[0] - tf.shape(rnn_out)[0]
    pad = tf.tile(rnn_out[-1:, :, :], [pad_time, 1, 1])
    rnn_out = tf.concat([rnn_out, pad], axis=0)                                 # (max_time, batch_size, rnn_size)
    pad_ids = tf.tile(output_ids[-1:, :], [pad_time, 1])
    output_ids = tf.concat([output_ids, pad_ids], axis=0)                       # (max_time, batch_size)
    
    return output_ids, rnn_out, dec_states


def split_heads(x, num_heads):
    """Split channels (dimension 3) into multiple heads (becomes dimension 1).

    Args:
        x: a Tensor with shape [batch, length, channels]
        num_heads: an integer

    Returns:
        a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    old_shape = x.get_shape().as_list()
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [num_heads] \
                + [last // num_heads if last else None]
    return tf.transpose(tf.reshape(x, new_shape), [0, 2, 1, 3])


def combine_heads(x):
    """Inverse of split_heads.

    Args:
        x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

    Returns:
        a Tensor with shape [batch, length, channels]
    """
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().as_list()
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    return tf.reshape(x, new_shape)


###############################################################################


class MultiDense(Dense):
    """
    Multi-dense layer.
    """
    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        output_shape = shape[:-1] + [self.units]
        
        if len(output_shape) > 2:
            inputs = tf.reshape(inputs, [-1, shape[-1]])
        inputs = tf.expand_dims(inputs, axis=2)
        kernel = tf.expand_dims(self.kernel, axis=0)
        # [?, num_units, input_dim]
        outputs = tf.transpose(inputs * kernel, [0, 2, 1])
        # [?, num_heads, num_units, input_dim / 4]
        outputs = split_heads(outputs, 4)
        # [?, num_heads, num_units]
        outputs = tf.reduce_sum(outputs, axis=3)
        # [?, num_units]
        outputs = tf.reduce_mean(outputs, axis=1)
        if len(output_shape) > 2:
            outputs = tf.reshape(outputs, output_shape)
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class SoftAttentionV3(attention_wrapper._BaseAttentionMechanism):
    """
    Implements Toronto-style (Xu et al.) attention scoring,
    as described in:
    "Show, Attend and Tell: Neural Image Caption Generation with 
    Visual Attention." ICML 2015. https://arxiv.org/abs/1502.03044
    
    and also multi-head attention.
    """
    # TODO: bookmark
    def __init__(self,
                 num_units,
                 feature_map,
                 fm_projection,
                 num_heads=None,
                 scale=True,
                 probability_fn=None,
                 name="SoftAttention"):
        """
        Construct the AttentionMechanism mechanism.
        Args:
            num_units: The depth of the attention mechanism.
            feature_map: The feature map / memory to query. This tensor
                should be shaped `[batch_size, height * width, channels]`.
            attention_type: String from 'single', 'multi_add', 'multi_dot'.
            reuse_keys_as_values: Boolean, whether to use keys as values.
            fm_projection: Feature map projection mode.
            num_heads: Int, number of attention heads. (optional)
            scale: Python boolean.  Whether to scale the energy term.
            probability_fn: (optional) A `callable`.  Converts the score
                to probabilities.  The default is `tf.nn.softmax`.
            name: Name to use when creating ops.
        """
        print("INFO: Using SoftAttentionV3.")
        assert fm_projection in [None, 'untied', 'tied']
        
        if probability_fn is None:
            probability_fn = tf.nn.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(SoftAttentionV3, self).__init__(
            query_layer=Dense(num_units, name="query_layer", use_bias=False),
            memory_layer=Dense(num_units, name="memory_layer", use_bias=False),     # self._keys is projected feature_map
            memory=feature_map,                                                     # self._values is feature_map
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=None,
            score_mask_value=float("-inf"),
            name=name)
        
        self._fm_projection = fm_projection
        self._num_units = num_units
        self._num_heads = num_heads
        self._scale = scale
        self._feature_map_shape = feature_map.get_shape().as_list()
        self._name = name
        
        if fm_projection == 'tied':
            self._values = split_heads(self._keys, self._num_heads)
        elif fm_projection == 'untied':
            # Project and split memory
            v_layer = Dense(num_units, name="value_layer", use_bias=False)
            # (batch_size, num_heads, max_time, num_units / num_heads)
            self._values = split_heads(v_layer(self._values), self._num_heads)
        else:
            self._values = split_heads(self._values, self._num_heads)
    
    
    def __call__(self, query, previous_alignments):
        """
        Score the query based on the keys and values.
        Args:
            query: RNN hidden state. Tensor of shape `[batch_size, num_units]`.
            previous_alignments: IGNORED. Previous alignment values.
                (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        with tf.variable_scope(None, "multi_add_attention", [query]):
            # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
            proj_query = tf.expand_dims(self.query_layer(query), 1)
            v = tf.get_variable(
              "attention_v", [self._num_units], dtype=proj_query.dtype)
            score = self._keys + proj_query
            shape = shape_list(score)
            score = normalise_activate('LN_tanh',
                                       tf.reshape(score, [-1, shape[2]]),
                                       tf.nn.tanh)
            score = v * tf.reshape(score, shape)                    # (batch_size, max_time, num_units)
            score = split_heads(score, self._num_heads)             # (batch_size, num_heads, max_time, num_units / num_heads)
            score = tf.reduce_sum(score, axis=3)                    # (batch_size, num_heads, max_time)
        
        if self._scale:
            softmax_temperature = tf.get_variable(
                    "softmax_temperature",
                    shape=[],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(5.0),
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                 "softmax_temperatures"])
            score /= softmax_temperature
        alignments = self._probability_fn(score, previous_alignments)
        return alignments
    '''
    @property
    def state_size(self):
        state = super(SoftAttentionV3, self).state_size()
        return state.clone(alignments=())
    '''
    
    def initial_alignments(self, batch_size, dtype):
        """Creates the initial alignment values for the `AttentionWrapper` class.
        
        This is important for AttentionMechanisms that use the previous alignment
        to calculate the alignment at the next time step (e.g. monotonic attention).
        
        The default behavior is to return a tensor of all zeros.
        
        Args:
            batch_size: `int32` scalar, the batch_size.
            dtype: The `dtype`.

        Returns:
            A `dtype` tensor shaped `[batch_size, alignments_size]`
            (`alignments_size` is the values' `max_time`).
        """
        return ()


class DeepOutputWrapper(rnn_cell_impl.RNNCell):
    """
    Wraps another `RNNCell` with attention, similar to `AttentionWrapper`,
    but performs optional deep output calculation (without the logits
    projection layer).
    
    Logits projection should be performed at the decoder by passing in
    an instance of `tf.layers.Layer`, as argument for `output_layer`.
    """
    
    def __init__(self,
                 cell,
                 initial_cell_state,
                 deep_output_layer=True,
                 name=None):
        super(DeepOutputWrapper, self).__init__(name=name)
        if not rnn_cell_impl._like_rnncell(cell):
            raise TypeError(
                    "cell must be an RNNCell, saw type: %s" % type(cell).__name__)
        self._cell = cell
        self._initial_cell_state = initial_cell_state
        self._deep_output_layer = deep_output_layer
    
    
    @property
    def output_size(self):
        return self._cell.output_size
    
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    
    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            return cell_state
    
    
    def __call__(self, inputs, state):
        """
        Perform a step of RNN with optional deep output layer.
        """
        cell_output, next_cell_state = self._cell(inputs, state)
        
        if self._deep_output_layer:
            # Deep output layer
            inputs_shape = shape_list(inputs)
            with tf.variable_scope("deep_output_layer"):
                cell_output = tf.reshape(cell_output, [-1, self.output_size])
                cell_output = linear("cell_output_projection",
                                     cell_output,
                                     inputs_shape[1],
                                     bias_init=None)
                inputs = tf.reshape(inputs, [-1, inputs_shape[1]])
                cell_output = cell_output + inputs
            cell_output = normalise_activate("output_projection",
                                             cell_output,
                                             tf.nn.tanh)
        return cell_output, next_cell_state


class AttentionDeepOutputWrapper(attention_wrapper.AttentionWrapper):
    """
    Wraps another `RNNCell` with attention, similar to `AttentionWrapper`,
    but performs deep output calculation (without logits projection layer).
    
    Logits projection should be performed at the decoder by passing in
    an instance of `tf.layers.Layer`, as argument for `output_layer`.
    """
    
    def call(self, inputs, state):
        """
        Perform a step of attention-wrapped RNN.
        
        This method assumes `inputs` is the word embedding vector.
        
        This method overrides the original `call()` method.
        """
        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        
        cell_batch_size = (
                cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory (encoder output) "
                "and the query (decoder output). Are you using the "
                "BeamSearchDecoder? You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                        [tf.assert_equal(cell_batch_size,
                                         self._attention_mechanism.batch_size,
                                         message=error_message)]):
            cell_output = tf.identity(cell_output, name="checked_cell_output")
        
        alignments = self._attention_mechanism(
                        cell_output, previous_alignments=state.alignments)
        
        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = tf.expand_dims(alignments, 1)
        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # alignments shape is
        #     [batch_size, 1, memory_time]
        # attention_mechanism.values shape is
        #     [batch_size, memory_time, attention_mechanism.num_units]
        # the batched matmul is over memory_time, so the output shape is
        #     [batch_size, 1, attention_mechanism.num_units].
        # we then squeeze out the singleton dim.
        attention_mechanism_values = self._attention_mechanism.values
        context = tf.matmul(expanded_alignments, attention_mechanism_values)
        context = tf.squeeze(context, [1])
        
        if self._attention_layer is not None:
            attention = self._attention_layer(
                    tf.concat([cell_output, context], 1))
        else:
            attention = context
        
        if self._alignment_history:
            alignment_history = state.alignment_history.write(
                    state.time, alignments)
        else:
            alignment_history = ()
        
        next_state = attention_wrapper.AttentionWrapperState(
                            time=state.time + 1,
                            cell_state=next_cell_state,
                            attention=attention,
                            alignments=alignments,
                            alignment_history=alignment_history)
        
        if self._output_attention:
            return attention, next_state
        else:
            # Deep output layer
            inputs_shape = shape_list(inputs)
            with tf.variable_scope("deep_output_layer"):
                cell_output = tf.reshape(cell_output, [-1, self.output_size])
                cell_output = linear("cell_output_projection",
                                     cell_output,
                                     inputs_shape[1],
                                     bias_init=None)
                inputs = tf.reshape(inputs, [-1, inputs_shape[1]])
                cell_output = cell_output + inputs
            cell_output = normalise_activate("output_projection",
                                             cell_output,
                                             tf.nn.tanh)
            return cell_output, next_state


class AttentionDeepOutputWrapperV2(attention_wrapper.AttentionWrapper):
    """
    Wraps another `RNNCell` with attention, similar to `AttentionWrapper`,
    but performs optional deep output calculation (without the logits
    projection layer).
    
    Logits projection should be performed at the decoder by passing in
    an instance of `tf.layers.Layer`, as argument for `output_layer`.
    """
    
    def __init__(self, deep_output_layer=True, **kwargs):
        super(AttentionDeepOutputWrapperV2, self).__init__(**kwargs)
        self._deep_output_layer = deep_output_layer
    
    
    def call(self, inputs, state):
        """
        Perform a step of attention-wrapped RNN.
        
        This method assumes `inputs` is the word embedding vector.
        
        This method overrides the original `call()` method.
        """
        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        
        cell_batch_size = (
                cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory (encoder output) "
                "and the query (decoder output). Are you using the "
                "BeamSearchDecoder? You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                        [tf.assert_equal(cell_batch_size,
                                         self._attention_mechanism.batch_size,
                                         message=error_message)]):
            cell_output = tf.identity(cell_output, name="checked_cell_output")
        
        alignments = self._attention_mechanism(
                        cell_output, previous_alignments=state.alignments)
        
        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = tf.expand_dims(alignments, 1)
        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # alignments shape is
        #     [batch_size, 1, memory_time]
        # attention_mechanism.values shape is
        #     [batch_size, memory_time, attention_mechanism.num_units]
        # the batched matmul is over memory_time, so the output shape is
        #     [batch_size, 1, attention_mechanism.num_units].
        # we then squeeze out the singleton dim.
        attention_mechanism_values = self._attention_mechanism.values
        context = tf.matmul(expanded_alignments, attention_mechanism_values)
        context = tf.squeeze(context, [1])
        
        if self._attention_layer is not None:
            attention = self._attention_layer(
                    tf.concat([cell_output, context], 1))
        else:
            attention = context
        
        if self._alignment_history:
            alignment_history = state.alignment_history.write(
                    state.time, alignments)
        else:
            alignment_history = ()
        
        next_state = attention_wrapper.AttentionWrapperState(
                            time=state.time + 1,
                            cell_state=next_cell_state,
                            attention=attention,
                            alignments=alignments,
                            alignment_history=alignment_history)
        
        if self._deep_output_layer:
            # Deep output layer
            inputs_shape = shape_list(inputs)
            with tf.variable_scope("deep_output_layer"):
                #cell_output = tf.reshape(cell_output, [-1, self.output_size])
                cell_output = linear("cell_output_projection",
                                     cell_output,
                                     inputs_shape[1],
                                     bias_init=None)
                inputs = tf.reshape(inputs, [-1, inputs_shape[1]])
                cell_output = cell_output + inputs
            cell_output = normalise_activate("output_projection",
                                             cell_output,
                                             tf.nn.tanh)
        return cell_output, next_state


class AttentionDeepOutputWrapperV3(attention_wrapper.AttentionWrapper):
    """
    Wraps another `RNNCell` with attention, similar to `AttentionWrapper`.
    Performs optional deep output calculation (without the logits
    projection layer).
    Allows optional multi-head attention.
    
    Logits projection should be performed at the decoder by passing in
    an instance of `tf.layers.Layer`, as argument for `output_layer`.
    """
    # TODO: bookmark
    def __init__(self,
                 deep_output_layer=True,
                 **kwargs):
        print("INFO: Using AttentionDeepOutputWrapperV3.")
        super(AttentionDeepOutputWrapperV3, self).__init__(**kwargs)
        self._deep_output_layer = deep_output_layer
    
    
    def call(self, inputs, state):
        """
        Perform a step of attention-wrapped RNN.
        
        This method assumes `inputs` is the word embedding vector.
        
        This method overrides the original `call()` method.
        """
        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        
        cell_batch_size = (
                cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory (encoder output) "
                "and the query (decoder output). Are you using the "
                "BeamSearchDecoder? You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                        [tf.assert_equal(cell_batch_size,
                                         self._attention_mechanism.batch_size,
                                         message=error_message)]):
            cell_output = tf.identity(cell_output, name="checked_cell_output")
        
        alignments = self._attention_mechanism(
                        cell_output, previous_alignments=())
        
        if len(shape_list(alignments)) == 3:
            # Multi-head attention
            expanded_alignments = tf.expand_dims(alignments, 2)
            # alignments shape is
            #     [batch_size, num_heads, 1, memory_time]
            # attention_mechanism.values shape is
            #     [batch_size, num_heads, memory_time, num_units / num_heads]
            # the batched matmul is over memory_time, so the output shape is
            #     [batch_size, num_heads, 1, num_units / num_heads].
            # we then combine the heads
            #     [batch_size, 1, attention_mechanism.num_units]
            attention_mechanism_values = self._attention_mechanism.values
            context = tf.matmul(expanded_alignments, attention_mechanism_values)
            attention = tf.squeeze(combine_heads(context), [1])
        else:
            # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
            expanded_alignments = tf.expand_dims(alignments, 1)
            # Context is the inner product of alignments and values along the
            # memory time dimension.
            # alignments shape is
            #     [batch_size, 1, memory_time]
            # attention_mechanism.values shape is
            #     [batch_size, memory_time, attention_mechanism.num_units]
            # the batched matmul is over memory_time, so the output shape is
            #     [batch_size, 1, attention_mechanism.num_units].
            # we then squeeze out the singleton dim.
            attention_mechanism_values = self._attention_mechanism.values
            context = tf.matmul(expanded_alignments, attention_mechanism_values)
            attention = tf.squeeze(context, [1])
        
        if self._alignment_history:
            '''
            alignment_history = state.alignment_history.write(
                    state.time, tf.reshape(
                alignments, [-1, self._attention_mechanism._alignments_size]))
            '''
            alignment_history = state.alignment_history.write(
                                                    state.time, alignments)
        else:
            alignment_history = ()
        
        next_state = attention_wrapper.AttentionWrapperState(
                            time=state.time + 1,
                            cell_state=next_cell_state,
                            attention=attention,
                            alignments=(),
                            alignment_history=alignment_history)
        
        if self._deep_output_layer:
            # Deep output layer
            inputs_shape = shape_list(inputs)
            with tf.variable_scope("deep_output_layer"):
                #cell_output = tf.reshape(cell_output, [-1, self.output_size])
                cell_output = linear("output_projection",
                                     cell_output,
                                     inputs_shape[1],
                                     bias_init=None)
                inputs = tf.reshape(inputs, [-1, inputs_shape[1]])
                cell_output = cell_output + inputs
                cell_output = normalise_activate("output_projection",
                                                 cell_output,
                                                 tf.nn.tanh)
        return cell_output, next_state
    
    
    @property
    def state_size(self):
        state = super(AttentionDeepOutputWrapperV3, self).state_size
        state = state.clone(alignments=())
        if self._attention_mechanism._fm_projection is None:
            state = state.clone(attention=self._attention_mechanism._feature_map_shape[-1])
        else:
            state = state.clone(attention=self._attention_mechanism._num_units)
        return state
    
    
    def zero_state(self, batch_size, dtype):
        state = super(AttentionDeepOutputWrapperV3, self).zero_state(
                                                        batch_size, dtype)
        #state = state.clone(alignments=())
        if self._attention_mechanism._fm_projection is None:
            state = state.clone(attention=tf.zeros(
                [batch_size, self._attention_mechanism._feature_map_shape[-1]], dtype))
        else:
            state = state.clone(attention=tf.zeros(
                [batch_size, self._attention_mechanism._num_units], dtype))
        return state


class BeamSearchDecoder_v2(tf.contrib.seq2seq.BeamSearchDecoder):
    def _merge_batch_beams(self, t, s=None):
        t_shape = t.get_shape().as_list()
        if len(t_shape) > 3:
            return tf.reshape(t, [-1, t_shape[2], t_shape[-1]])
        else:
            return super(BeamSearchDecoder_v2, self)._merge_batch_beams(t, s)
    
    
    def _split_batch_beams(self, t, s=None):
        t_shape = t.get_shape().as_list()
        if len(t_shape) > 2:
            shape = [self._batch_size, self._beam_width, t_shape[1], -1]
            return tf.reshape(t, shape)
        else:
            return super(BeamSearchDecoder_v2, self)._split_batch_beams(t, s)


