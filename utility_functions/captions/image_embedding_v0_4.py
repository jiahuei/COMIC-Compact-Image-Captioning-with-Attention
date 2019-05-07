# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:43:32 2017

@author: jiahuei

Image embedding ops.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from utility_functions.captions import slim_nets
slim = tf.contrib.slim


def image_model(config,
                images,
                spatial_squeeze=True,
                is_training=True):
    """Builds an image model subgraph for image embeddings.
    
    Args:
        images: A float32 Tensor of shape [batch, height, width, channels].
        spatial_squeeze: Whether to remove dimensions of size 1.
        is_training: Boolean indicating training mode or not.
    
    Returns:
        end_points: A dictionary of activations from image model layers.
    """
    assert config.image_model in ['vgg_16',
                                  'InceptionV1',
                                  'InceptionV3',
                                  'InceptionV4']
    is_model_training = config.train_image_model and is_training
    
    if is_model_training:
        keep_prob = 1 - config.dropout_im
    else:
        keep_prob = 1.0
    
    if config.image_model == 'vgg_16':
        with slim.arg_scope(slim_nets.vgg_arg_scope(weight_decay=0.0)):
            if config.vgg_endpoint == 'fc7':
                keep_prob = math.sqrt(keep_prob)
            net_output, end_points = slim_nets.vgg_16(
                                            images,
                                            is_training=is_model_training,
                                            dropout_keep_prob=keep_prob,
                                            final_endpoint=config.vgg_endpoint,
                                            spatial_squeeze=spatial_squeeze)
    elif config.image_model == 'InceptionV1':
        with slim.arg_scope(slim_nets.inception_v1_arg_scope(
                                weight_decay=0.0,
                                use_batch_norm=True)):
            net_output, end_points = slim_nets.inception_v1(
                                            images,
                                            is_training=is_model_training,
                                            dropout_keep_prob=keep_prob,
                                            spatial_squeeze=spatial_squeeze)
    elif config.image_model == 'InceptionV3':
        with slim.arg_scope(slim_nets.inception_v3_arg_scope(
                                weight_decay=0.0)):
            net_output, end_points = slim_nets.inception_v3(
                                            images,
                                            is_training=is_model_training,
                                            dropout_keep_prob=keep_prob,
                                            spatial_squeeze=spatial_squeeze)
    else:
        with slim.arg_scope(slim_nets.inception_v4_arg_scope(
                                weight_decay=0.0)):
            net_output, end_points = slim_nets.inception_v4(
                                            images,
                                            is_training=is_model_training,
                                            dropout_keep_prob=keep_prob,
                                            spatial_squeeze=spatial_squeeze)
    return net_output, end_points


