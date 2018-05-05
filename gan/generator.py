# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Style transfer network code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from magenta.models.image_stylization import ops

slim = tf.contrib.slim


def transform(input_, normalizer_fn=ops.conditional_instance_norm,
              normalizer_params=None, reuse=False):
  """Maps content images to stylized images.

  Args:
    input_: Tensor. Batch of input images.
    normalizer_fn: normalization layer function.  Defaults to
        ops.conditional_instance_norm.
    normalizer_params: dict of parameters to pass to the conditional instance
        normalization op.
    reuse: bool. Whether to reuse model parameters. Defaults to False.

  Returns:
    Tensor. The output of the transformer network.
  """
  if normalizer_params is None:
    normalizer_params = {'center': True, 'scale': True}
  with tf.variable_scope('transformer', reuse=reuse):
    with slim.arg_scope(
        [slim.conv2d],
        activation_fn=tf.nn.relu,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params,
        weights_initializer=tf.random_normal_initializer(0.0, 0.01),
        biases_initializer=tf.constant_initializer(0.0)):
      with tf.variable_scope('contract'):
        h = conv2d(input_, 9, 1, 32, 'conv1')
        h = conv2d(h, 3, 2, 64, 'conv2')
        h = conv2d(h, 3, 2, 128, 'conv3')
      with tf.variable_scope('residual'):
        h = residual_block(h, 3, 'residual1')
        h = residual_block(h, 3, 'residual2')
        h = residual_block(h, 3, 'residual3')
        h = residual_block(h, 3, 'residual4')
        h = residual_block(h, 3, 'residual5')
      with tf.variable_scope('expand'):
        h = upsampling(h, 3, 2, 64, 'conv1')
        h = upsampling(h, 3, 2, 32, 'conv2')
        return upsampling(h, 9, 1, 3, 'conv3', activation_fn=tf.nn.sigmoid)


def conv2d(input_,
           kernel_size,
           stride,
           num_outputs,
           scope,
           activation_fn=tf.nn.relu):
  """Same-padded convolution with mirror padding instead of zero-padding.

  This function expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor input.
    kernel_size: int (odd-valued) representing the kernel size.
    stride: int representing the strides.
    num_outputs: int. Number of output feature maps.
    scope: str. Scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  padding = kernel_size // 2
  padded_input = tf.pad(
      input_, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
      mode='REFLECT')
  return slim.conv2d(
      padded_input,
      padding='VALID',
      kernel_size=kernel_size,
      stride=stride,
      num_outputs=num_outputs,
      activation_fn=activation_fn,
      scope=scope)


def upsampling(input_,
               kernel_size,
               stride,
               num_outputs,
               scope,
               activation_fn=tf.nn.relu):
  """A smooth replacement of a same-padded transposed convolution.

  This function first computes a nearest-neighbor upsampling of the input by a
  factor of `stride`, then applies a mirror-padded, same-padded convolution.

  It expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor input.
    kernel_size: int (odd-valued) representing the kernel size.
    stride: int representing the strides.
    num_outputs: int. Number of output feature maps.
    scope: str. Scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  with tf.variable_scope(scope):
    shape = tf.shape(input_)
    height = shape[1]
    width = shape[2]
    upsampled_input = tf.image.resize_nearest_neighbor(
        input_, [stride * height, stride * width])
    return conv2d(
        upsampled_input,
        kernel_size,
        1,
        num_outputs,
        'conv',
        activation_fn=activation_fn)


def residual_block(input_, kernel_size, scope, activation_fn=tf.nn.relu):
  """A residual block made of two mirror-padded, same-padded convolutions.

  This function expects `kernel_size` to be odd.

  Args:
    input_: 4-D Tensor, the input.
    kernel_size: int (odd-valued) representing the kernel size.
    scope: str, scope under which to operate.
    activation_fn: activation function.

  Returns:
    4-D Tensor, the output.

  Raises:
    ValueError: if `kernel_size` is even.
  """
  if kernel_size % 2 == 0:
    raise ValueError('kernel_size is expected to be odd.')
  with tf.variable_scope(scope):
    num_outputs = input_.get_shape()[-1].value
    h_1 = conv2d(input_, kernel_size, 1, num_outputs, 'conv1', activation_fn)
    h_2 = conv2d(h_1, kernel_size, 1, num_outputs, 'conv2', None)
    return input_ + h_2

# @slim.add_arg_scope
def conditional_instance_norm(inputs,
                              labels,
                              num_categories,
                              center=True,
                              scale=True,
                              activation_fn=None,
                              reuse=None,
                              variables_collections=None,
                              outputs_collections=None,
                              trainable=True,
                              scope=None):
  """Conditional instance normalization from TODO(vdumoulin): add link.

    "A Learned Representation for Artistic Style"

    Vincent Dumoulin, Jon Shlens, Manjunath Kudlur

  Can be used as a normalizer function for conv2d.

  Args:
    inputs: a tensor with 4 dimensions. The normalization occurs over height
        and width.
    labels: tensor, style labels to condition on.
    num_categories: int, total number of styles being modeled.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: Optional activation function.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined, or if the
        input doesn't have 4 dimensions.
  """
  with tf.variable_scope(scope, 'InstanceNorm', [inputs],
                         reuse=reuse) as sc:
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    if inputs_rank != 4:
      raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = [1, 2]
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))