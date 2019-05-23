# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities for preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inputs.preprocessing import inception_preprocessing_v3 as inception_prepro
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import time, numpy as np

slim = tf.contrib.slim
distort_crop = inception_prepro.distorted_bounding_box_crop
distort_colour_inception = inception_prepro.distort_color
random_apply = inception_prepro.apply_with_random_selector

_PADDING = 2

'''
def preprocess_image(image, output_height, output_width, is_training):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  image = tf.to_float(image)
  image = tf.image.resize_image_with_crop_or_pad(
      image, output_width, output_height)
  image = tf.subtract(image, 128.0)
  image = tf.div(image, 128.0)
  return image
'''

def _elastic_transform(x, alpha=None, sigma=None,
                       mode="constant", cval=0, is_random=False):
    """Elastic transformation for image as described in `[Simard2003] <http://deeplearning.cs.cmu.edu/pdfs/Simard.pdf>`__.
    Parameters
    -----------
    x : numpy.array
        A greyscale image.
    alpha : float
        Alpha value for elastic transformation.
    sigma : float or sequence of float
        The smaller the sigma, the more transformation. Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.
    mode : str
        See `scipy.ndimage.filters.gaussian_filter <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html>`__. Default is `constant`.
    cval : float,
        Used in conjunction with `mode` of `constant`, the value outside the image boundaries.
    is_random : boolean
        Default is False.
    Returns
    -------
    numpy.array
        A processed image.
    Examples
    ---------
    >>> x = tl.prepro.elastic_transform(x, alpha=x.shape[1]*3, sigma=x.shape[1]*0.07)
    References
    ------------
    - `Github <https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a>`__.
    - `Kaggle <https://www.kaggle.com/pscion/ultrasound-nerve-segmentation/elastic-transform-for-data-augmentation-0878921a>`__
    """
    if is_random is False:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(int(time.time()))
    
    if alpha is None:
        alpha = x.shape[1] * 4.0
    if sigma is None:
        sigma = x.shape[1] * 0.10
    
    is_3d = False
    if len(x.shape) == 3 and x.shape[-1] == 1:
        x = x[:, :, 0]
        is_3d = True
    elif len(x.shape) == 3 and x.shape[-1] != 1:
        raise Exception("Only support greyscale image")
    assert len(x.shape) == 2, "input should be grey-scale image"

    shape = x.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma,
                         mode=mode, cval=cval) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma,
                         mode=mode, cval=cval) * alpha
    
    x_, y_ = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                         indexing='ij')
    indices = np.reshape(x_ + dx, (-1, 1)), np.reshape(y_ + dy, (-1, 1))
    
    if is_3d:
        return map_coordinates(x, indices, order=1).reshape((shape[0], shape[1], 1))
    else:
        return map_coordinates(x, indices, order=1).reshape(shape)


def _elastic_transform_tf(image):
    with tf.name_scope('elastic_transform'):
        image = tf.py_func(lambda x: _elastic_transform(x), [image], tf.float32)
    return image


def _random_shear_tf(image):
    fn = lambda x: tf.keras.preprocessing.image.random_shear(
                            x, intensity=30,
                            row_axis=0, col_axis=1, channel_axis=2,
                            fill_mode='constant')
    with tf.name_scope('random_shear'):
        image = tf.py_func(fn, [image], tf.float32)
    return image


def _distort_color(image, color_ordering=0, scope=None):
    """Distort the color of a Tensor image.
    
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    
    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-1).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 1]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=63. / 255.)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        else:
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            image = tf.image.random_brightness(image, max_delta=63. / 255.)
    return image


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         padding=_PADDING,
                         fast_mode=True,
                         add_image_summaries=True):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        padding: The amound of padding before and after each dimension of the image.
        add_image_summaries: Enable image summaries.

    Returns:
        A preprocessed image.
    """
    if add_image_summaries:
        tf.summary.image('image', tf.expand_dims(image, 0))
    
    if output_height == output_width == 28:
        output_channel = 1                      # MNIST
    else:
        output_channel = 3                      # CIFAR
    
    # Transform the image to floats.
    image = tf.to_float(image)
    
    if padding > 0:
        image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
    
    # Resize to 40 x 40
    distorted_image = tf.image.resize_images(
            image, size=[40, 40], method=tf.image.ResizeMethod.BILINEAR)
    
    if not fast_mode:
        # Random rotation
        with tf.name_scope('random_rotation'):
            rand = tf.random_uniform([], minval=-0.8, maxval=0.8)   # 45 deg
            distorted_image = tf.contrib.image.rotate(
                                    images=distorted_image,
                                    angles=rand,
                                    interpolation='BILINEAR')
        # Elastic transform
        distorted_image = _elastic_transform_tf(distorted_image)
    
    # Randomly crop a [height, width] section of the image.
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32,
                       shape=[1, 1, 4])
    
    distorted_image, distorted_bbox = distort_crop(
                            distorted_image, bbox, area_range=(0.75, 1.0))
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, output_channel])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
                        tf.expand_dims(image, 0), distorted_bbox)
    if add_image_summaries:
        tf.summary.image('images_with_distorted_bounding_box',
                         image_with_distorted_box)
    
    # Resize to final size
    num_resize_cases = 1 if fast_mode else 4
    distorted_image = random_apply(
        distorted_image,
        lambda x, method: tf.image.resize_images(
                                x, [output_height, output_width], method),
        num_cases=num_resize_cases)
    
    if output_channel == 3:
        # Randomly flip the image horizontally for CIFAR.
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # Random colour augment
        distorted_image = random_apply(
                distorted_image,
                lambda x, ordering: distort_colour_inception(x, ordering),
                num_cases=2 if fast_mode else 4)
        rand = tf.random_uniform([], minval=0.1, maxval=0.5)
    else:
        # Random colour augment
        distorted_image = random_apply(
                distorted_image,
                lambda x, ordering: _distort_color(x, ordering),
                num_cases=2)
        rand = tf.random_uniform([], minval=0.1, maxval=1.0)
    
    # Add noise
    distorted_image += tf.random_normal(
            [output_height, output_width, output_channel], stddev=rand)
    #distorted_image = tf.clip_by_value(distorted_image, 0.0, 1.0)              # causes NaN loss for some reason
    
    # Subtract off the mean and divide by the variance of the pixels.
    distorted_image = tf.image.per_image_standardization(distorted_image)
    
    if add_image_summaries:
        tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))
    
    return distorted_image


def preprocess_for_eval(image, output_height, output_width,
                        add_image_summaries=True):
    """Preprocesses the given image for evaluation.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        add_image_summaries: Enable image summaries.

    Returns:
        A preprocessed image.
    """
    if add_image_summaries:
        tf.summary.image('image', tf.expand_dims(image, 0))
    
    # Transform the image to floats.
    image = tf.to_float(image)
    
    # Resize and crop if needed.
    resized_image = tf.image.resize_image_with_crop_or_pad(image,
                                                           output_width,
                                                           output_height)
    if add_image_summaries:
        tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))
    
    # Subtract off the mean and divide by the variance of the pixels.
    return tf.image.per_image_standardization(resized_image)


def preprocess_image(image, output_height, output_width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True,
                     add_image_summaries=True):
    """Preprocesses the given image.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        is_training: `True` if we're preprocessing the image for training and
            `False` otherwise.
        add_image_summaries: Enable image summaries.

    Returns:
        A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(
                image, output_height, output_width,
                fast_mode=fast_mode, add_image_summaries=add_image_summaries)
    else:
        return preprocess_for_eval(
                image, output_height, output_width,
                add_image_summaries=add_image_summaries)





