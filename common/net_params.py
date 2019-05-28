# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:21:36 2019

@author: jiahuei

Network parameters, preprocessing functions, etc.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

pjoin = os.path.join


all_net_params = dict(
                vgg_16 = dict(
                    name = 'vgg_16',
                    ckpt_path = 'vgg_16.ckpt',
                    url = 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
                    ),
                resnet_v1_50 = dict(
                    name = 'resnet_v1_50',
                    ckpt_path = 'resnet_v1_50.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
                    ),
                resnet_v1_101 = dict(
                    name = 'resnet_v1_101',
                    ckpt_path = 'resnet_v1_101.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
                    ),
                resnet_v1_152 = dict(
                    name = 'resnet_v1_152',
                    ckpt_path = 'resnet_v1_152.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
                    ),
                resnet_v2_50 = dict(
                    name = 'resnet_v2_50',
                    ckpt_path = 'resnet_v2_50.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
                    ),
                resnet_v2_101 = dict(
                    name = 'resnet_v2_101',
                    ckpt_path = 'resnet_v2_101.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
                    ),
                resnet_v2_152 = dict(
                    name = 'resnet_v2_152',
                    ckpt_path = 'resnet_v2_152.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
                    ),
                inception_v1 = dict(
                    name = 'inception_v1',
                    ckpt_path = 'inception_v1.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
                    ),
                inception_v2 = dict(
                    name = 'inception_v2',
                    ckpt_path = 'inception_v2.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz',
                    ),
                inception_v3 = dict(
                    name = 'inception_v3',
                    ckpt_path = 'inception_v3.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
                    ),
                inception_v4 = dict(
                    name = 'inception_v4',
                    ckpt_path = 'inception_v4.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
                    ),
                inception_resnet_v2 = dict(
                    name = 'inception_resnet_v2',
                    ckpt_path = 'inception_resnet_v2_2016_08_30.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz',
                    ),
                )


def get_net_params(net_name):
    net_params = all_net_params[net_name]
    base_dir = os.path.split(os.path.dirname(__file__))[0]
    net_params['ckpt_path'] = pjoin(base_dir, 'ckpt', net_params['ckpt_path'])
    return net_params



