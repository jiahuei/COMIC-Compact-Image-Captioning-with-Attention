# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:59:29 2017

@author: jiahuei
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys
import json, string
import random
from nets import nets_factory
from inputs.preprocessing import preprocessing_factory as prepro_factory
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CURR_DIR))
import ops
_shape = ops.shape
pjoin = os.path.join
slim = tf.contrib.slim


class InputManager(object):
    """ Input Manager object."""
    
    def __init__(self, config, is_inference=False):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """ 
        print('INFO: Using `manager_image_caption_v1`.')
        #super(InputManager, self).__init__(config, is_inference)
        c = config
        
        # Determine the input size of image CNN
        im_net = nets_factory.networks_map[c.cnn_name]
        s = c.cnn_input_size
        if isinstance(s, list) and len(s) == 2 and 0 not in s:
            print('INFO: Using specified CNN input size: {}.'.format(s))
        else:
            if hasattr(im_net, 'default_image_size'):
                c.cnn_input_size = [im_net.default_image_size] * 2
                print('INFO: Using default CNN input size: {}.'.format(
                                                        c.cnn_input_size))
            else:
                raise ValueError('Unable to retrieve default image size.')
        self._setup(c, is_inference)
    
    
    def _setup(self, config, is_inference):
        # Add new info to config
        config.split_sizes = {}
        self.config = c = config
        self.is_inference = is_inference
        random.seed(c.rand_seed)
        
        # Read vocab files
        self._get_vocab()
        
        # Read test set filenames
        if is_inference:
            if 'coco' in c.infer_set:
                if c.infer_set == 'coco_test':
                    coco_set = 'test2014'
                else:
                    coco_set = 'val2014'
                    c.batch_size_infer = 61
                fname_list = os.listdir(pjoin(c.dataset_dir, coco_set))
                self.filenames_infer = [pjoin(c.dataset_dir, coco_set, ff)
                                        for ff in fname_list]
            else:
                if c.infer_set == 'test':
                    fname_list = 'filenames_test.txt'
                elif c.infer_set == 'valid':
                    fname_list = 'filenames_valid.txt'
                with open(pjoin(c.dataset_dir, 'captions', fname_list)) as f:
                    self.filenames_infer = [l.strip() for l in f.readlines()]
        
        # Bucketing and batching
        if 'coco' in c.dataset_file_pattern:
            self.buckets = [11, 13, 15]      # MSCOCO word-based
        elif 'insta' in c.dataset_file_pattern:
            self.buckets = [7, 10, 13]
        
        # Setup input pipelines
        with tf.device("/cpu:0"):
            if is_inference:
                self.batch_infer = self._batch_setup('infer')
            else:
                self.batch_train = self._batch_setup('train')
                self.batch_eval = self._batch_setup('valid')
        print("INFO: Input pipelines setup complete.")
    
    
    def _get_vocab(self):
        c = self.config
        if '{}' not in c.dataset_file_pattern:
            raise ValueError('`dataset_file_pattern` must have `{}`.')
        fp = pjoin(c.dataset_dir, 'captions', c.dataset_file_pattern.format('itow'))
        with open(fp + '.json', 'r') as f:
            self.config.itow = json.load(f)
        fp = pjoin(c.dataset_dir, 'captions', c.dataset_file_pattern.format('wtoi'))
        with open(fp + '.json', 'r') as f:
            self.config.wtoi = json.load(f)
        self.config.vocab_size = len(self.config.itow)
    
    
    def _batch_setup(self, split):
        """
        Produce a batch of examples using tf.data.
        """
        c = self.config
        
        is_training = 'train' in split and not self.is_inference
        if self.is_inference:
            batch_size = c.batch_size_infer
            data = []
            for f in self.filenames_infer:
                data.append([f, ['null']])
            assert len(data) % batch_size == 0
            self.config.split_sizes['infer'] = len(data)
        else:
            # Data format: filepath,w0 w1 w2 w3 w4 ... wN
            fp = pjoin(c.dataset_dir, 'captions',
                       c.dataset_file_pattern.format(split))
            with open(fp + '.txt', 'r') as f:
                data = [l.strip().split(',') for l in f.readlines()]
            data = [[l[0], l[1].split(' ')] for l in data]
            self.config.split_sizes[split] = len(data)
            
            if is_training:
                #num_threads = max(4, c.num_threads)
                try:
                    gs = c.accum_grads_step
                except:
                    gs = 1
                batch_size = c.batch_size_train
                self.config.max_step = int(len(data) / batch_size * c.max_epoch / gs)
            else:
                #num_threads = 1
                batch_size = c.batch_size_eval
                assert len(data) % batch_size == 0
        
        with tf.name_scope('batch_{}'.format(split)):
            
            def _wrap(prepro_fn, *args):
                """ Wraps the image preprocessing / augmenting function. """
                return lambda im, cap: (prepro_fn(im, *args), cap)
            
            im_size = c.cnn_input_size
            augment = is_training and c.cnn_input_augment
            print('INFO: Augment {} images: {}'.format(split, augment))
            im_prepro_fn = prepro_factory.get_preprocessing(
                                        c.cnn_name, is_training=augment)
            #im_prepro_fn = lambda im, cap: (
            #                im_prepro_fn(im, im_size[0], im_size[1]), cap)
            im_prepro_fn = _wrap(im_prepro_fn, im_size[0], im_size[1])
            
            # Fetch filepaths and captions from data generator
            dataset = tf.data.Dataset.from_generator(
                                generator=lambda: self._gen(data, is_training),
                                #generator=gen,
                                output_shapes=(None, [None]),
                                output_types=(tf.string, tf.int32))
            # Read the images
            dataset = dataset.map(self._read_image, num_parallel_calls=3)
            # Pre-fetch (~4x increase in training speed)
            dataset = dataset.prefetch(batch_size * 15)
            # Pre-process / Augment the images
            dataset = dataset.map(im_prepro_fn, num_parallel_calls=3)
            # Pre-fetch again
            dataset = dataset.prefetch(batch_size * 15)
                
            dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
                    element_length_func=lambda im, cap: tf.shape(cap)[0],
                    bucket_boundaries=self.buckets,
                    bucket_batch_sizes=[batch_size] * (len(self.buckets) + 1),
                    padded_shapes=None,
                    padding_values=(.0, c.wtoi['<PAD>']),
                    pad_to_bucket_boundary=False))
            dataset = dataset.map(
                    lambda im, cap: self._set_shape(batch_size, im, cap))
            # Get the dataset iterator
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            return batch
    
    
    def _read_image(self, fp, caption):
        image = tf.image.decode_image(tf.read_file(fp), channels=3)
        image.set_shape([None, None, 3])
        #image = tf.random_normal(shape=[224, 224, 3])
        return image, caption
    
    
    def _set_shape(self, batch_size, im, cap):
        im_size = _shape(im)
        im.set_shape([batch_size, im_size[0], im_size[1], 3])
        cap.set_shape([batch_size, None])
        return im, cap
    
    
    def _gen(self, data, is_training=True):
        """
        Generator fn, yields the image filepath and word IDs.
        Handles dataset shuffling.
        """
        c = self.config
        
        idx = 0
        if is_training:
            random.shuffle(data)
            print('INFO: Training data shuffled, idx {:3,d}'.format(idx))
        
        while True:
            for d in data:
                # d[0] is filepath, d[1] is a list of chars / words
                caption = [c.wtoi.get(w, c.wtoi['<UNK>']) for w in d[1]]
                caption = np.array(caption)
                yield (pjoin(c.dataset_dir, d[0]), caption.astype(np.int32))
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data)
                print('INFO: Training data shuffled, idx {:3,d}'.format(idx))


class InputManager_Radix(InputManager):
    """ Input Manager object for Radix-token models."""
    
    def __init__(self, config, is_inference=False):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """ 
        super(InputManager_Radix, self).__init__(config, is_inference)
        c = self.config
        max_word_len = len(ops.number_to_base(len(c.wtoi), c.radix_base))
        self.buckets = [b * max_word_len for b in self.buckets]
        self.radix_wtoi = {}
        assert c.wtoi['<PAD>'] == -1
        for k in c.wtoi:
            if k == '<GO>':
                idx = [c.radix_base]
            elif k == '<EOS>':
                idx = [c.radix_base + 1]
            elif k == '<PAD>':
                idx = [-1]
            else:
                idx = ops.number_to_base(c.wtoi[k], c.radix_base)
                idx = [0] * (max_word_len - len(idx)) + idx
            self.radix_wtoi[k] = idx
        #self.config.radix_wtoi = self.radix_wtoi
    
    
    def _gen(self, data, is_training=True):
        """
        Generator fn, yields the image filepath and word IDs.
        Handles dataset shuffling.
        """
        c = self.config
        
        idx = 0
        if is_training:
            random.shuffle(data)
            print('INFO: Training data shuffled, idx {:3,d}'.format(idx))
        
        while True:
            for d in data:
                # d[0] is filepath, d[1] is a list of chars / words
                caption = [self.radix_wtoi.get(w, self.radix_wtoi['<UNK>']) 
                                                    for w in d[1]]
                caption = np.concatenate(caption)
                yield (pjoin(c.dataset_dir, d[0]), caption.astype(np.int32))
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data)
                print('INFO: Training data shuffled, idx {:3,d}'.format(idx))


class InputManager_Char(InputManager):
    """ Input Manager object for character-token models."""
    
    def __init__(self, config, is_inference=False):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """ 
        super(InputManager_Char, self).__init__(config, is_inference)
        c = self.config
        if 'coco' in c.dataset_file_pattern:
            self.buckets = [45, 55, 70]
        elif 'insta' in c.dataset_file_pattern:
            self.buckets = [29, 42, 61]
    
    
    def _get_vocab(self):
        c = self.config
        if '{}' not in c.dataset_file_pattern:
            raise ValueError('`dataset_file_pattern` must have `{}`.')
        
        fp = pjoin(c.dataset_dir, 'captions', c.dataset_file_pattern.format('wtoi'))
        with open(fp + '.json', 'r') as f:
            wtoi = json.load(f)
        pad_value = wtoi['<PAD>']
        char_list = list(string.digits + string.ascii_lowercase)
        
        ctoi = {}
        itoc = {}
        idx = pad_value
        ctoi['<PAD>'] = idx
        itoc[idx] = '<PAD>'
        idx += 1
        ctoi[' '] = idx
        itoc[idx] = ' '
        idx += 1
        
        for c in char_list:
            ctoi[c] = idx
            itoc[idx] = c
            idx += 1
        ctoi['<GO>'] = len(ctoi)
        ctoi['<EOS>'] = len(ctoi)
        itoc[len(itoc)] = '<GO>'
        itoc[len(itoc)] = '<EOS>'
        
        self.config.itow = itoc
        self.config.wtoi = ctoi
        self.config.vocab_size = len(self.config.itow)
    
    
    def _gen(self, data, is_training=True):
        """
        Generator fn, yields the image filepath and word IDs.
        Handles dataset shuffling.
        """
        c = self.config
        
        idx = 0
        if is_training:
            random.shuffle(data)
            print('INFO: Training data shuffled, idx {:3,d}'.format(idx))
        
        while True:
            for d in data:
                # d[0] is filepath, d[1] is a list of chars / words
                caption = [c.wtoi[ch] for ch in ' '.join(d[1][1:-1])]
                caption = [c.wtoi['<GO>']] + caption + [c.wtoi['<EOS>']]
                caption = np.array(caption)
                yield (pjoin(c.dataset_dir, d[0]), caption.astype(np.int32))
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data)
                print('INFO: Training data shuffled, idx {:3,d}'.format(idx))


class InputManager_SCST(InputManager_Radix):
    """ Input Manager object."""
    
    def __init__(self, config, is_inference=False):
        """
        Loads the h5 file containing caption data and corresponding image paths.
        """ 
        super(InputManager_SCST, self).__init__(config, is_inference)
    
    
    def _batch_setup(self, split):
        """
        Produce a batch of examples using tf.data.
        """
        c = self.config
        
        is_training = 'train' in split and not self.is_inference
        if self.is_inference:
            batch_size = c.batch_size_infer
            data = []
            for f in self.filenames_infer:
                data.append([f, ['null']])
            assert len(data) % batch_size == 0
            self.config.split_sizes['infer'] = len(data)
        else:
            # Data format: filepath,w0 w1 w2 w3 w4 ... wN
            fp = pjoin(c.dataset_dir, 'captions',
                       c.dataset_file_pattern.format(split))
            
            with open(fp + '.txt', 'r') as f:
                data = [l.strip().split(',') for l in f.readlines()]
            data_dict = {}
            for d in data:
                if d[0] not in data_dict:
                    data_dict[d[0]] = []
                s = d[1].replace('<GO> ', '').replace(' <EOS>', '')
                data_dict[d[0]].append(s)
            
            data = data_dict.items()
            del(data_dict)
            self.config.split_sizes[split] = len(data)
            
            if is_training:
                #num_threads = max(4, c.num_threads)
                try:
                    gs = c.accum_grads_step
                except:
                    gs = 1
                batch_size = c.batch_size_train
                self.config.max_step = int(len(data) / batch_size * c.max_epoch / gs)
            else:
                return None
        
        with tf.name_scope('batch_{}'.format(split)):
            
            def _wrap(prepro_fn, *args):
                """ Wraps the image preprocessing / augmenting function. """
                return lambda im, cap: (prepro_fn(im, *args), cap)
            
            im_size = c.cnn_input_size
            augment = is_training and c.cnn_input_augment
            print('INFO: Augment {} images: {}'.format(split, augment))
            im_prepro_fn = prepro_factory.get_preprocessing(
                                        c.cnn_name, is_training=augment)
            #im_prepro_fn = lambda im, cap: (
            #                im_prepro_fn(im, im_size[0], im_size[1]), cap)
            im_prepro_fn = _wrap(im_prepro_fn, im_size[0], im_size[1])
            
            # Fetch filepaths and captions from data generator
            dataset = tf.data.Dataset.from_generator(
                                generator=lambda: self._gen(data, is_training),
                                #generator=gen,
                                output_shapes=(None, [None]),
                                output_types=(tf.string, tf.string))
            # Read the images
            dataset = dataset.map(self._read_image, num_parallel_calls=3)
            # Pre-fetch (~4x increase in training speed)
            dataset = dataset.prefetch(batch_size * 15)
            # Pre-process / Augment the images
            dataset = dataset.map(im_prepro_fn, num_parallel_calls=3)
            # Pre-fetch again
            dataset = dataset.prefetch(batch_size * 15)
                
            dataset = dataset.apply(
                        tf.contrib.data.batch_and_drop_remainder(batch_size))
            # Get the dataset iterator
            iterator = dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            return batch
    
    
    def _gen(self, data, is_training=True):
        """
        Generator fn, yields the image filepath and word IDs.
        Handles dataset shuffling.
        """
        c = self.config
        
        idx = 0
        if is_training:
            random.shuffle(data)
            print('INFO: Training data shuffled, idx {:3,d}'.format(idx))
        
        while True:
            for d in data:
                # d[0] is filepath, d[1] is a list of captions for that image
                # for MSCOCO Karpathy split, there are 113,287 train images
                #     308 (0.272 %) have more than 5 GT captions
                #     for convenience, we just take 5 captions
                yield (pjoin(c.dataset_dir, d[0]), np.array(d[1][:5]))
                idx += 1
            # Shuffle at the end of epoch
            if is_training:
                random.shuffle(data)
                print('INFO: Training data shuffled, idx {:3,d}'.format(idx))
    
    
    def captions_to_batched_ids(self, hypos):
        """
        Generates batched IDs with padding for SCST training.
        Used as GT for XE objective.
        """
        c = self.config
        assert c.token_type in ['radix', 'word', 'char']
        
        hypos_idx = []
        for h in hypos:
            if c.token_type == 'radix':
                h = ['<GO>'] + h[0].split() + ['<EOS>']
                h = [self.radix_wtoi.get(w, self.radix_wtoi['<UNK>']) for w in h]
                h = np.concatenate(h)
            elif c.token_type == 'word':
                h = ['<GO>'] + h[0].split() + ['<EOS>']
                h = [c.wtoi.get(w, c.wtoi['<UNK>']) for w in h]
                h = np.array(h)
            elif c.token_type == 'char':
                h = [c.wtoi[ch] for ch in h[0]]
                h = [c.wtoi['<GO>']] + h + [c.wtoi['<EOS>']]
                h = np.array(h)
            hypos_idx.append(h)
        
        assert len(hypos_idx[0].shape) == 1
        max_hypo_len = max([hy.shape[0] for hy in hypos_idx])
        assert max_hypo_len > 1
        for i, h in enumerate(hypos_idx):
            h = np.pad(h, pad_width=[0, max_hypo_len - len(h)],
                       mode='constant', constant_values=c.wtoi['<PAD>'])
            hypos_idx[i] = h
        hypos_idx = np.stack(hypos_idx, axis=0)
        return hypos_idx






