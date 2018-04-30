import sys
import os
import numpy as np
sys.path.append('../../magenta/magenta/models/arbitrary_image_stylization/')
sys.path.append('../../magenta/magenta/models/image_stylization/')
from image_utils import *

def load_xy_pair(fake_directory, real_directory, prob_of_real = 0.5): 
    '''
    Loads a pair of tensors that are the x (input) and y (label) to the 
    discriminator network. Randomly picks a "fake" or "real" image 

    Input: 
    fake_directory: A string filepath to the directory containig "fake" images
    real_directory: A string filepath to the directory containig "real" images
    prob_of_real: Float in [0, 1]. Probability of selecting a "real" image. 

    Output: 
    img: A tensor of shape [1, ?, ?, 3]
    y: A tensor that is either [1, 0] (real) or [0, 1] (fake)
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

def gen_labels(is_real = True, batch_size = 4): 
    if is_real: 
        return tf.constant(np.repeat(np.array([[1.0, 0.0]]), batch_size, axis=0))
    else: 
        return tf.constant(np.repeat(np.array([[0.0, 1.0]]), batch_size, axis=0))