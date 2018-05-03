import sys
import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
sys.path.append('../magenta/magenta/models/arbitrary_image_stylization/')
sys.path.append('../magenta/magenta/models/image_stylization/')

from magenta.models.image_stylization import learning as learning_utils
from magenta.models.image_stylization import vgg
from data_processing_utils import * 
from arbitrary_image_stylization_losses import * 

# Imports for the model build (forward pass). 
from magenta.models.arbitrary_image_stylization import nza_model as transformer_model
from magenta.models.image_stylization import ops
from tensorflow.contrib.slim.python.slim.nets import inception_v3

class GAN_model(object):
	"""
	Based on an online tutorial: 

	"""
	
	def __init__(self, mode):
		"""Basic setup.
		
		Args:
			mode: "train" or "generate"
		"""
		assert mode in ["train", "generate"]
		self.mode = mode
		
		# set up hyperparameters ..

		# # hyper-parameters for model
		# self.x_dim = 784
		# self.z_dim = 100
		self.batch_size = FLAGS.batch_size
		# self.num_samples = FLAGS.num_samples
		
		# TODO is global step useful for us? 
		# maybe for cmputing test loss? 

		# Global step Tensor.
		self.global_step = None
		
		print('The mode is %s.' % self.mode)
		print('complete initializing model.')
		
		
	def build_inputs(self):
		# TODO REPLACE WITH OUR NEEDED PLACEHOLDERS

		# with tf.variable_scope('random_z'):
		# 	self.random_z = tf.placeholder(tf.float32, [None, self.z_dim])

		# return self.random_z
	
	def stylize_images(content_input, style_input, trainable, is_training, reuse=None, 
		inception_end_point='Mixed_6e', style_prediction_bottleneck=100, adds_losses=True, 
		content_weights=None, style_weights=None, total_variation_weight=None):
		'''
		See build_model.
		'''
		# def build_model(content_input_,
		  #               style_input_,
		  #               trainable,
		  #               is_training,
		  #               reuse=None,
		  #               inception_end_point='Mixed_6e',
		  #               style_prediction_bottleneck=100,
		  #               adds_losses=True,
		  #               content_weights=None,
		  #               style_weights=None,
		  #               total_variation_weight=None):

  		# Call style_transfer_losses to get all the different losses! 
  		[activation_names,
		   activation_depths] = transformer_model.style_normalization_activations()

		  # Defines the style prediction network.
		  # From the build_model file import style_prediction
		style_params, bottleneck_feat = style_prediction(
		  style_input_,
		  activation_names,
		  activation_depths,
		  is_training=is_training,
		  trainable=trainable,
		  inception_end_point=inception_end_point,
		  style_prediction_bottleneck=style_prediction_bottleneck,
		  reuse=reuse)

		# Defines the style transformer network.
		stylized_images = transformer_model.transform(
		  content_input_,
		  normalizer_fn=ops.conditional_style_norm,
		  reuse=reuse,
		  trainable=trainable,
		  is_training=is_training,
		  normalizer_params={'style_params': style_params})

		return None 

  
	def style_transfer_losses(content_inputs, style_inputs, stylized_inputs, reuse=tf.AUTO_REUSE): 
		"""Computes the total loss function.

		The total loss function is composed of a content, a style and a total
		variation term.

		Args:
		  content_inputs: Tensor. The input images.
		  style_inputs: Tensor. The input images.
		  stylized_inputs: Tensor. The stylized input images.
		  content_weights: dict mapping layer names to their associated content loss
		      weight. Keys that are missing from the dict won't have their content
		      loss computed.
		  style_weights: dict mapping layer names to their associated style loss
		      weight. Keys that are missing from the dict won't have their style
		      loss computed.
		  total_variation_weight: float. Coefficient for the total variation part of
		      the loss.
		  reuse: bool. Whether to reuse model parameters. Defaults to False.

		Returns:
		  Tensor for the total loss, dict mapping loss names to losses.
	  	"""

		content_weights = {"vgg_16/conv3": 1}
		style_weights = {"vgg_16/conv1": 0.5e-3, "vgg_16/conv2": 0.5e-3,
                         "vgg_16/conv3": 0.5e-3, "vgg_16/conv4": 0.5e-3}
		total_variation_weight = 1e4                 

		with tf.name_scope('content_endpoints'):
			content_end_points = vgg.vgg_16(content_inputs, reuse=True)
		with tf.name_scope('style_endpoints'):
			style_end_points = vgg.vgg_16(style_inputs, reuse=True)
		with tf.name_scope('stylized_endpoints'):
			stylized_end_points = vgg.vgg_16(stylized_inputs, reuse=True)

			# Compute the content loss
		with tf.name_scope('content_loss'):
			total_content_loss, content_loss_dict = arbitrary_image_stylization_losses.content_loss(
				content_end_points, stylized_end_points, content_weights)

			# Compute the style loss
		with tf.name_scope('style_loss'):
			total_style_loss, style_loss_dict = arbitrary_image_stylization_losses.style_loss(
				style_end_points, stylized_end_points, style_weights)

			# Compute the total variation loss
		with tf.name_scope('total_variation_loss'):
			tv_loss, total_variation_loss_dict = learning_utils.total_variation_loss(
				stylized_inputs, total_variation_weight)

			# Compute the total loss
		with tf.name_scope('total_loss'):
			loss = total_content_loss + total_style_loss + tv_loss

		loss_dict = {'total_loss': loss}
		loss_dict.update(content_loss_dict)
		loss_dict.update(style_loss_dict)
		loss_dict.update(total_variation_loss_dict)

		return [loss, total_content_loss, total_style_loss, tv_loss, loss_dict]

		


	def Discriminator(self, data, reuse=tf.AUTO_REUSE):
		# data = images we need to predict...

		with tf.variable_scope('Discriminator', reuse=reuse) as scope:
			conv1_out = slim.conv2d(data, 1, [5, 5], activation_fn=tf.nn.relu, scope='conv1')
			pool1_out = slim.max_pool2d(conv1_out, [2, 2], stride=2, scope='pool1')
			conv2_out = slim.conv2d(pool1_out, 1, [4, 4], stride=2, activation_fn=tf.nn.relu, scope='conv2')
			pool2_out = slim.max_pool2d(conv2_out, [6, 6], stride=3, padding='SAME', scope='pool2')
			flat_out = slim.flatten(pool2_out, scope='flatten3')
			fc1_out = slim.fully_connected(flat_out, 12, scope='fc1', activation_fn=tf.nn.relu)
			predictions_logits = slim.fully_connected(fc1_out, 2, activation_fn=tf.nn.softmax)
			return predictions_logits 
	
	def Discriminator_loss(predictions, labels):
		# predictions already softmax'd 
		# labels are [a, b] where a + b = 1 
		cross_entropy_loss = slim.losses.softmax_cross_entropy(predictions, labels)

		# compute accuracy, where prediction is argmax 
		pred_classes = tf.argmax(predictions, axis=1)
		true_classes = tf.argmax(labels, axis=1)
		accuracy = slim.metrics.accuracy(pred_classes, true_classes)

		# compute confidence in more-probable class 
		mean_confidence = tf.reduce_mean(tf.reduce_max(predictions, axis=1))

		return cross_entropy_loss
		
	def setup_global_step(self):
		"""Sets up the global step Tensor."""
		if self.mode == "train":
			self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False,
								           collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
			
			print('complete setup global_step.')
			
	
	
	

	# def GANLoss(self, logits, is_real=True, scope=None):
		# TODO IMPORT CONTENT, STYLE LOSS
		# AND DISCRIMINATOR LOSS


			
	def build(self):
		"""Creates all ops for training or generate."""
		self.setup_global_step()
		
		
		if self.mode == "generate":
			pass
		
		else:
			path_to_tripled_data = '/Users/akhiljalan/Downloads/traindata3/train0/'

			
			content_path = path_to_tripled_data + 'content/'
			style_path = path_to_tripled_data + 'style/'


			# TODO make content, style batch
			content_batch, style_batch = None, None

			# Compute two forward passes of the generator. 
			stylized_images = stylize_images(content_batch, style_batch)
			unstylized_images = stylize_images(stylized_images, content_batch)

			#def style_transfer_losses(content_inputs, style_inputs, stylized_inputs, reuse=tf.AUTO_REUSE): 
			# return loss, total_content_loss, total_style_loss, tv_loss, loss_dict
			transfer_1_losses = style_transfer_losses(content_batch, style_batch, stylized_images)
			transfer_2_losses = style_transfer_losses(stylized_images, content_batch, unstylized_images)
			

			
			fake_labels = data_processing_utils.gen_labels(is_real=False, batch_size=self.batch_size)

			# fake_images, fake_labels, fake_paths = get_xy_pairs(real_path, fake_path, self.batch_size, prob_of_real = 0.0)

			# Compute real discriminator logits, loss 
			real_images, real_labels, real_paths = get_xy_pairs(real_path, fake_path, self.batch_size, prob_of_real = 1.0)

			real_predictions = self.Discriminator(real_images)

			d_real_loss = self.Discriminator_loss(real_predictions, real_labels)

			# Compute fake discriminator logits, loss 
			fake_predictions = self.Discriminator(unstylized_images)
			d_fake_loss = self.Discriminator_loss(fake_predictions, fake_labels)

			# Compute generator-discriminator loss 
			gen_fooling_loss = self.Discriminator_loss(fake_predictions, real_labels)

			# Compute total losses
			# Generator losses
			with tf.variable_scope('loss_G'):
				self.loss_Generator = transfer_1_losses[0] + transfer_2_losses[0] + gen_fooling_loss

			# losses of Discriminator
			with tf.variable_scope('loss_D'):
				total_discr_loss = d_real_loss + d_fake_loss
				self.loss_Discriminator = total_discr_loss

			# TODO this might cause some bugs...

			self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
			self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
			
			
			# generating images for sample
			# self.sample_data = self.Generator(self.random_z, reuse=True)
			
			# write summaries
			# Add loss summaries
			# TODO add more fine-grained losses, i.e. content, style, discrim losses 
			tf.summary.scalar('losses/loss_Discriminator', self.loss_Discriminator)
			tf.summary.scalar('losses/loss_Generator', self.loss_Generator)
			

			#TODO 

			# Set up training
			optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
			train_op = slim.learning.create_train_op(
			  total_loss,
			  optimizer,
			  clip_gradient_norm=FLAGS.clip_gradient_norm,
			  summarize_gradients=True) #False? 

			# Function to restore VGG16 parameters.
			init_fn_vgg = slim.assign_from_checkpoint_fn(vgg.checkpoint_file(),
			                                           slim.get_variables('vgg_16'))

			# Function to restore Inception_v3 parameters.
			inception_variables_dict = {
			  var.op.name: var
			  for var in slim.get_model_variables('InceptionV3')
			}
			init_fn_inception = slim.assign_from_checkpoint_fn(
			  FLAGS.inception_v3_checkpoint, inception_variables_dict)

			# Function to restore VGG16 and Inception_v3 parameters.
			def init_sub_networks(session):
				init_fn_vgg(session)
				init_fn_inception(session)

			# Run training
			# slim.learning.train(
			#   train_op=train_op,
			#   logdir=os.path.expanduser(FLAGS.train_dir),
			#   master=FLAGS.master,
			#   is_chief=FLAGS.task == 0,
			#   number_of_steps=FLAGS.train_steps,
			#   init_fn=init_sub_networks,
			#   save_summaries_secs=FLAGS.save_summaries_secs,
			#   save_interval_secs=FLAGS.save_interval_secs)

			# Add histogram summaries
			# for var in self.D_vars:
			# 	tf.summary.histogram(var.op.name, var)
			# for var in self.G_vars:
			# 	tf.summary.histogram(var.op.name, var)
			
			# Add image summaries
			# tf.summary.image('random_images', tf.reshape(self.generated_data, [-1, 28, 28, 1]), max_outputs=4)
			#tf.summary.image('real_images', tf.reshape(self.real_data, [-1, 28, 28, 1]))
			
		print('complete model build.\n')
