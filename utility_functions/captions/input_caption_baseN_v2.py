#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 21:27:06 2017

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from .configuration import *
#from utility_functions.captions.configuration import Config
import h5py
import tensorflow as tf
import numpy as np
import os
import json
import cPickle as pickle
pjoin = os.path.join


def load_config(config_filepath):
    with open(config_filepath, 'rb') as f:
        config = pickle.load(f)
    return config


class _BaseInputManager(object):
    """ Base Input Manager object."""
    
    def __init__(self, config, mode=None):
        """
        Setups the necessary information.
        """ 
        self.train_data_splits = 4
        self.config = config
        self.mode = mode
        
        c = self.config
        if c.image_model in ['InceptionV3', 'InceptionV4']:
            # Inception-v3 image sizes
            self.resize_size = [346, 346]
            self.input_size = [299, 299]
        else:
            # Inception-v1, VGG image sizes
            self.resize_size = [256, 256]    # TODO: Maybe allow jittering [256, 512]
            self.input_size = [224, 224]
        
        with open(pjoin(c.data_root, c.itow_file), 'r') as f:
            self.config.itow = json.load(f)
        with open(pjoin(c.data_root, c.wtoi_file), 'r') as f:
            self.config.wtoi = json.load(f)


class InputManager(_BaseInputManager):
    """ Input Manager object."""
    
    def __init__(self, config, mode=None):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """ 
        super(InputManager, self).__init__(config, mode)
        
        train, valid, test = self._load_and_shuffle_data()
        
        # Get train, valid, test sizes
        self.dset_sizes = [[len(t['targets']) for t in train],
                           len(valid['targets']),
                           len(test['targets'])]
        self.dset_max_time = [[t['targets'].shape[1] for t in train],
                              valid['targets'].shape[1],
                              test['targets'].shape[1]]
        
        self.train = train
        self.valid = valid
        self.test = test
        self.test_set_filepaths = list(set(test['im_paths']))
        self.config.train_size = sum(self.dset_sizes[0])
        print("INFO: Data set loading complete.")
    
    
    def initialise_tf_queue(self):
        """ 
        Set ups the TF subgraphs responsible for feeding the main graph.
        """
        with tf.name_scope("input_queues"):
            with tf.device("/cpu:0"):
                self._init_tf_queue()
        print("INFO: Input queues setup complete.")
    
    
    def _load_and_shuffle_data(self):
        """
        Loads the h5 file containing captions and image paths,
        and shuffles them.
        """
        c = self.config
        train = []
        with h5py.File(pjoin(c.data_root, c.caption_file), 'r') as f:
            for i in range(self.train_data_splits):
                train.append(
                    dict(im_paths = list(f['train_%d/image_paths' % i][:]),
                         inputs = f['train_%d/inputs' % i][:],
                         targets = f['train_%d/targets' % i][:]))
            
            valid = dict(im_paths = list(f['valid/image_paths'][:]),
                         inputs = f['valid/inputs'][:],
                         targets = f['valid/targets'][:])
            
            test = dict(im_paths = list(f['test/image_paths'][:]),
                        inputs = f['test/inputs'][:],
                        targets = f['test/targets'][:])
        
        # Shuffle training data
        np.random.seed(c.rand_seed)
        
        for t in train:
            perm = np.arange(len(t['targets']))
            np.random.shuffle(perm)
            t['inputs'] = t['inputs'][perm, :]
            t['targets'] = t['targets'][perm, :]
            t['im_paths'] = [t['im_paths'][i] for i in perm]
        
        # Get full image paths
        for t in train:
            t['im_paths'] = [pjoin(c.data_root, p) for p in t['im_paths']]
        valid['im_paths'] = [pjoin(c.data_root, p) for p in valid['im_paths']]
        test['im_paths'] = [pjoin(c.data_root, p) for p in test['im_paths']]
        
        return train, valid, test
    
    
    def _init_tf_queue(self):
        """ 
        Set ups the TF subgraphs responsible for feeding the main graph.
        """
        capacity = int(self.config.capacity_mul_factor * self.config.batch_size)
        
        ### Prepare training batches ###
        
        train_batch_ops = []
        for i, t in enumerate(self.train):
            batch_ops = self._produce_batch(
                                        t['inputs'],
                                        t['targets'],
                                        t['im_paths'],
                                        capacity,
                                        None,
                                        self.config.batch_threads[0],
                                        'train_%d' % i)
            train_batch_ops.append(batch_ops)
        
        # Randomly get a batch of training data
        rand = tf.random_uniform([], minval=0, maxval=1.0)
        p0 = self.dset_sizes[0][0] / sum(self.dset_sizes[0])
        p1 = self.dset_sizes[0][1] / sum(self.dset_sizes[0]) + p0
        p2 = self.dset_sizes[0][2] / sum(self.dset_sizes[0]) + p1
        train_batch_compiled = tf.case(
                        [(tf.less(rand, p0), lambda: train_batch_ops[0]),
                         (tf.less(rand, p1), lambda: train_batch_ops[1]),
                         (tf.less(rand, p2), lambda: train_batch_ops[2])],
                        default=lambda: train_batch_ops[3])
        
        self.train_batch_ops = tf.train.batch(
                                train_batch_compiled,
                                batch_size=self.config.batch_size,
                                num_threads=self.config.batch_threads[0],
                                capacity=capacity * 5,
                                enqueue_many=True,
                                #shapes=[self.IMAGE_SIZE, [None], [None]],
                                dynamic_pad=True,
                                allow_smaller_final_batch=False,
                                name='train_batch_final')
        
        ### Prepare validation batches ###
        
        self.valid_batch_ops = self._produce_batch(
                                        self.valid['inputs'],
                                        self.valid['targets'],
                                        self.valid['im_paths'],
                                        capacity,
                                        None,
                                        self.config.batch_threads[1],
                                        'valid')
        
        ### Prepare testing batches ###
        
        self.test_batch_ops = self._produce_batch(
                                        self.test['inputs'],
                                        self.test['targets'],
                                        self.test['im_paths'],
                                        capacity,
                                        None,
                                        self.config.batch_threads[2],
                                        'test')
    
    
    def _produce_batch(self,
                       dec_inputs,
                       dec_targets,
                       file_list,
                       capacity,
                       num_epochs,
                       num_threads,
                       name):
        """
        Produce mini-batches from captions and images.
        """
        c = self.config
        seed = c.rand_seed
        batch_size = c.batch_size
        
        # Read images
        image = self._produce_images(
                                file_list,
                                capacity,
                                num_epochs,
                                name)
        
        # Read captions
        dec_inputs, dec_targets = self._produce_captions(
                                            dec_inputs,
                                            dec_targets,
                                            capacity,
                                            name)
        
        # Produce mini-batches
        capacity *= 5
        min_after_dequeue = int(0.5 * capacity)
        #shapes = [self.feature_size, self.cnn_fm_size, [max_time], [max_time]]
        
        batch_op = tf.train.shuffle_batch(
                                [dec_inputs, dec_targets, image],
                                batch_size=batch_size,
                                capacity=capacity,
                                min_after_dequeue=min_after_dequeue,
                                num_threads=num_threads,
                                seed=seed,
                                enqueue_many=False,
                                #shapes=shapes,
                                allow_smaller_final_batch=False,
                                name="%s_batch" % name)
        return batch_op
    
    
    def _produce_images(self,
                        file_list,
                        capacity,
                        num_epochs,
                        name):
        """
        Produce image tensor by reading JPG files.
        """
        file_queue = tf.train.string_input_producer(
                                        file_list, 
                                        num_epochs=num_epochs, 
                                        shuffle=False,
                                        seed=None,
                                        capacity=capacity,
                                        name="%s_image_producer" % name)
        reader = tf.WholeFileReader()
        im_path, value = reader.read(file_queue)
        im_path = tf.reshape(im_path, [-1])
        image = tf.image.decode_image(value, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, self.resize_size)
        image = tf.squeeze(image)
        
        # Crop to final dimensions.
        if 'train' in name:
            # Randomly distort the image.
            image = self._distort_image(image)
        else:
            # Central crop.
            image = tf.image.resize_image_with_crop_or_pad(
                                image, self.input_size[0], self.input_size[1])
        image.set_shape(self.input_size + [3])
        return image
    
    
    def _produce_captions(self,
                          dec_inputs,
                          dec_targets,
                          capacity,
                          name):
        """
        Produce captions row by row from the matrix.
        """
        dec_inputs = tf.train.input_producer(
                                    dec_inputs, 
                                    num_epochs=None,
                                    shuffle=False,
                                    seed=None,
                                    capacity=capacity,
                                    name="%s_dec_inputs_producer" % name)
        dec_targets = tf.train.input_producer(
                                    dec_targets, 
                                    num_epochs=None,
                                    shuffle=False,
                                    seed=None,
                                    capacity=capacity,
                                    name="%s_dec_targets_producer" % name)
        dec_inputs = tf.to_int32(dec_inputs.dequeue())
        dec_targets = tf.to_int32(dec_targets.dequeue())
        return dec_inputs, dec_targets
    
    
    def _distort_image(self, image):
        """Perform random distortions on an image.
    
        Args:
            image: A float32 Tensor of shape [height, width, 3]
                with values in [0, 1).
    
        Returns:
            distorted_image: A float32 Tensor of shape [height, width, 3]
                with values in [0, 1].
        """
        c = self.config
        
        def _crop_only(image):
            image = tf.random_crop(image, self.input_size + [3])
            return image
        
        def _distort_A(image):
            image = tf.random_crop(image, self.input_size + [3])
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.020)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            return image
        
        def _distort_B(image):
            image = tf.random_crop(image, self.input_size + [3])
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.010)
            return image
        
        # Randomly flip horizontally.
        with tf.name_scope("flip_horizontal", values=[image]):
            image = tf.image.random_flip_left_right(image)
        
        if c.distort_images:
            # Randomly distort the colors.
            with tf.name_scope("distort_color", values=[image]):
                rand = tf.random_uniform([], minval=0, maxval=1.0)
                image = tf.case(
                        [(tf.less(rand, 0.475), lambda: _distort_A(image)),     # 47.5 % chance
                         (tf.less(rand, 0.950), lambda: _distort_B(image))],    # 95.0 % chance
                        default=lambda: _crop_only(image))                      #  5.0 % chance
                
                # The random_* ops do not necessarily clamp.
                image = tf.clip_by_value(image, 0.0, 1.0)
        else:
            image = tf.random_crop(image, self.input_size + [3])
        return image


class InputManagerInfer(InputManager):
    """ Input Manager object for inference."""
    
    def __init__(self,
                 config,
                 filepaths):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """ 
        super(InputManagerInfer, self).__init__(config)
        
        # Get the list of images
        self.test_set_filenames = [os.path.basename(f) for f in filepaths]
        self.test_set_filepaths = filepaths
        self.config.test_size = len(self.test_set_filepaths)
        
        c = self.config
        assert c.test_size % c.batch_size == 0, \
            "`test_size` of %d not divisible by `batch_size` of %d." % \
                (c.test_size, c.batch_size)
    
    
    def initialise_tf_queue(self):
        """ 
        Set ups the TF subgraphs responsible for feeding the main graph.
        """
        with tf.name_scope("input_queues"):
            with tf.device("/cpu:0"):
                self._init_test_queue()
        print("INFO: Input queues for test data setup complete.")
    
    
    def _init_test_queue(self):
        """ 
        Set ups the TF subgraphs responsible for feeding the main graph.
        """
        c = self.config
        capacity = int(c.capacity_mul_factor * c.batch_size)
        
        ### Prepare testing batches ###
        image = self._produce_images(
                                self.test_set_filepaths,
                                capacity=capacity,
                                num_epochs=1,
                                name='test')
        
        self.test_batch_ops = tf.train.batch(
                                        [image],
                                        batch_size=c.batch_size,
                                        num_threads=1,
                                        capacity=capacity * 2,
                                        enqueue_many=False,
                                        #shapes=self.IMAGE_SIZE,
                                        dynamic_pad=False,
                                        allow_smaller_final_batch=False,
                                        name='test_batch')
        
        
        
