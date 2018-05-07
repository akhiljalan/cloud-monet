from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import tempfile


import numpy as np
from scipy.misc import imread
import tensorflow as tf


def load_xy_pair(fake_directory, real_directory, prob_of_real = 0.5): 
    '''
    Loads a pair of tensors that are the x (input) and y (label) to the 
    discriminator network. Randomly picks a "fake" or "real" image. 

    Note that setting prob_of_real = 0.0 guarantees a fake image, while setting
    prob_of_real = 1.0 guarantees a real image. 

    Input: 
    fake_directory: A string filepath to the directory containig "fake" images
    real_directory: A string filepath to the directory containig "real" images
    prob_of_real: Float in [0, 1]. Probability of selecting a "real" image. 

    Output: 
    img: A tensor of shape [1, ?, ?, 3]
    y: A tensor of shape [1, 2]. It is either [1, 0] (real) or [0, 1] (fake).
    '''
    coin_flip = np.random.binomial(1, prob_of_real)
    fake_img_names = os.listdir(fake_directory)
    real_img_names = os.listdir(real_directory)
    name = ''
    if coin_flip == 1: 
        # pick real 
        index = np.random.randint(low=0, high=len(real_img_names))
        name = real_img_names[index]
        img_path = real_directory + name
        y = tf.constant([[1.0, 0.0]])
    else: 
        # pick fake
        index = np.random.randint(low=0, high=len(fake_img_names))
        name = fake_img_names[index]
        img_path = fake_directory + name
        y = tf.constant([[0.0, 1.0]])
    img = load_image(img_path, image_size=256)
    return img, y, name
        
def load_xy_pairs(fake_directory, real_directory, batch_size = 4, prob_of_real = 0.5): 
    '''
    Loads a batch of tensor pairs (x, y) that are the x (input) and y (label) to the 
    discriminator network. Randomly picks a "fake" or "real" image each time. 
    See "load_xy_pair" 
    
    Input: 
    fake_directory: A string filepath to the directory containig "fake" images
    real_directory: A string filepath to the directory containig "real" images
    batch_size: A positive integer. Number of tensor pairs to load up. 
    prob_of_real: Float in [0, 1]. Probability of selecting a "real" image. 

    Output: 
    images: A tensor of shape [batch_size, ?, ?, 3]
    labels: A tensor of shape [batch_size, 2]. 
    '''
    fake_img_names = os.listdir(fake_directory)
    real_img_names = os.listdir(real_directory)
    images, labels, first_img_path = load_xy_pair(fake_directory, real_directory, prob_of_real)
    img_paths = [first_img_path]
    for i in range(batch_size - 1): 
        new_img, new_label, img_path = load_xy_pair(fake_directory, real_directory, prob_of_real)
        images = tf.concat([images, new_img], axis = 0)
        labels = tf.concat([labels, new_label], axis = 0)
        img_paths.append(img_path)
    return images, labels, img_paths

def load_random_images(filepath, batch_size = 4): 
    img_names = os.listdir(filepath)

    # Create a set of random indices from [0, ..., len(img_names) - 1]
    indices = np.random.choice(np.arange(len(img_names)), batch_size, replace=False)

    # Load a single random image. 
    images = load_image(filepath + img_names[indices[0]], image_size=256)

    # Load the rest of the images, concatenating onto the images tensor. 
    for i in range(1, len(indices)): 
        index = indices[i]
        next_img = load_image(filepath + img_names[index], image_size=256)
        images = tf.concat([images, next_img], axis=0)
    return images

def gen_labels(is_real = True, batch_size = 4): 
    if is_real: 
        return tf.constant(np.repeat(np.array([[1.0, 0.0]]), batch_size, axis=0))
    else: 
        return tf.constant(np.repeat(np.array([[0.0, 1.0]]), batch_size, axis=0))

def load_np_image_uint8(image_file):
    """Loads an image as a numpy array.
    Source: Google Magenta (magenta/models/imgage_stylization/image_utils.py)

    Args:
    image_file: str. Image file.

    Returns:
    A 3-D numpy array of shape [image_size, image_size, 3] and dtype uint8,
    with values in [0, 255].
    """
    with tempfile.NamedTemporaryFile() as f:
        f.write(tf.gfile.GFile(image_file, 'rb').read())
        f.flush()
        image = imread(f.name)
        # Workaround for black-and-white images
        if image.ndim == 2:
            image = np.tile(image[:, :, None], (1, 1, 3))
    return image

def load_np_image(image_file):
    """Loads an image as a numpy array.
    Source: Google Magenta (magenta/models/imgage_stylization/image_utils.py)

    Args:
    image_file: str. Image file.

    Returns:
    A 3-D numpy array of shape [image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
    """
    return np.float32(load_np_image_uint8(image_file) / 255.0)

def load_image(image_file, image_size=None):
    """Loads an image and center-crops it to a specific size.
    Source: Google Magenta (magenta/models/imgage_stylization/image_utils.py)

    Args:
    image_file: str. Image file.
    image_size: int, optional. Desired size. If provided, crops the image to
    a square and resizes it to the requested size. Defaults to None.

    Returns:
    A 4-D tensor of shape [1, image_size, image_size, 3] and dtype float32,
    with values in [0, 1].
    """
    image = tf.constant(np.uint8(load_np_image(image_file) * 255.0))
    if image_size is not None:
    # Center-crop into a square and resize to image_size
        small_side = min(image.get_shape()[0].value, image.get_shape()[1].value)
        image = tf.image.resize_image_with_crop_or_pad(image, small_side, small_side)
        image = tf.image.resize_images(image, [image_size, image_size])
        image = tf.to_float(image) / 255.0
    return tf.expand_dims(image, 0)
