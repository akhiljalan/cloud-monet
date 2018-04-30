import tensorflow as tf
import tensorflow.contrib.eager as tfe

slim = tf.contrib.slim
import sys
import os
import numpy as np
from data_processing_utils import * 

train_real_path = '/Users/akhiljalan/Downloads/train0/content_train/'
train_fake_path = '/Users/akhiljalan/Downloads/train0/identity_train/'
test_real_path = '/Users/akhiljalan/Downloads/train0/content_test/'
test_fake_path = '/Users/akhiljalan/Downloads/train0/identity_test/'


def discriminator_network(images, is_training=True): 
	with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
		conv1_out = slim.conv2d(images, 1, [5, 5], activation_fn=tf.nn.relu, scope='conv1')
		pool1_out = slim.max_pool2d(conv1_out, [2, 2], stride=2, scope='pool1')
		conv2_out = slim.conv2d(pool1_out, 1, [4, 4], stride=2, activation_fn=tf.nn.relu, scope='conv2')
		pool2_out = slim.max_pool2d(conv2_out, [6, 6], stride=3, padding='SAME', scope='pool2')
		flat_out = slim.flatten(pool2_out, scope='flatten3')
		fc1_out = slim.fully_connected(flat_out, 12, scope='fc1', activation_fn=tf.nn.relu)
		predictions = slim.fully_connected(fc1_out, 2, activation_fn=tf.nn.softmax)
		return predictions

def evalaute_discriminator(inputs, labels, img_paths, step_num, num_evals=1):
	

	predictions = discriminator_network(inputs)

	cross_entropy_loss = slim.losses.softmax_cross_entropy(predictions, labels)

	# compute accuracy, where prediction is argmax 
	pred_classes = tf.argmax(predictions, axis=1)
	true_classes = tf.argmax(labels, axis=1)
	accuracy = slim.metrics.accuracy(pred_classes, true_classes)

	# compute confidence in more-probable class 
	mean_confidence = tf.reduce_mean(tf.reduce_max(predictions, axis=1))

	# create summary logs 
	tf.summary.scalar('test_metrics/loss', cross_entropy_loss)
	tf.summary.scalar('test_metrics/acc', accuracy)
	tf.summary.scalar('test_metrics/confidence', mean_confidence)

	metric_dict = {
		'test/loss': cross_entropy_loss,
		'test/acc': accuracy,
		'test/confidence': mean_confidence
	}

	metrics = {}
	for key, value in metric_dict.iteritems():
		metrics[key] = tf.metrics.mean(value)

	names_values, names_updates = slim.metrics.aggregate_metric_map(metrics)
	for name, value in names_values.iteritems():
		slim.summaries.add_scalar_summary(value, name, print_summary=True)
	eval_op = names_updates.values()

	# tf.summary.image('test_images/image_{}'.format(img_paths), images, 3)

	# https://resources.oreilly.com/examples/0636920079057/blob/d70c450721e852c1b1df90369bf8a0d1fafb2d3c/Training,%20Evaluating,%20and%20Tuning%20Deep%20Neural%20Network%20Models%20with%20TensorFlow-Slim%20-%20Working%20Files/Chapter%202/utils/slim_training_evaluation.py
	# Define the metrics:
	# names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
	# 	'test/loss': cross_entropy_loss,
	# 	'test/acc': accuracy,
	# 	'test/confidence': mean_confidence
	# })

	

	print('Running evaluation Loop...')

	checkpoint_path = tf.train.latest_checkpoint('./logdir/')

	metric_values = slim.evaluation.evaluate_once(
		num_evals=num_evals,
		master='',
		checkpoint_path=checkpoint_path,
		logdir=checkpoint_path,
		eval_op=eval_op)
		# final_op=names_to_values.values())

	# names_to_values = dict(zip(names_values.keys(), metric_values))
	# for name in names_to_values:
	# 	print('%s: %f' % (name, names_to_values[name]))


def main(): 
	# get global step 
	# tf_global_step = tf.train.get_or_create_global_step()

	with tf.Graph().as_default() as graph: 
		# Set params up. 
		tf.logging.set_verbosity(tf.logging.DEBUG)

		global_step = tf.train.get_or_create_global_step(graph=graph)
		
		# Load Data
		images, labels, img_paths = load_xy_pairs(train_fake_path, train_real_path, batch_size=2)

		# Make a forward pass. 
		predictions = discriminator_network(images, is_training=True)

		# Define losses, accuracy, confidence.
		cross_entropy_loss = slim.losses.softmax_cross_entropy(predictions, labels)

		# compute accuracy, where prediction is argmax 
		pred_classes = tf.argmax(predictions, axis=1)
		true_classes = tf.argmax(labels, axis=1)
		accuracy = slim.metrics.accuracy(pred_classes, true_classes)

		# compute confidence in more-probable class 
		mean_confidence = tf.reduce_mean(tf.reduce_max(predictions, axis=1))

		# create summary logs 
		tf.summary.scalar('train_metrics/loss', cross_entropy_loss)
		tf.summary.scalar('train_metrics/acc', accuracy)
		tf.summary.scalar('train_metrics/confidence', mean_confidence)
		tf.summary.image('train_images/image_{}'.format(img_paths), images, 3)
		# summary_op = tf.summary.merge_all()

		#TODO TEST EVAL 
		# https://www.tensorflow.org/api_docs/python/tf/contrib/metrics/aggregate_metric_map
		test_images, test_labels, test_img_paths = load_xy_pairs(test_fake_path, test_real_path, batch_size=50)

		if global_step % 100 == 0: 
			print('hahahaha')
			evalaute_discriminator(test_images, test_labels, test_img_paths, global_step)


		


		# misc ops 
		optimizer = tf.train.AdamOptimizer(1e-4)
		train_op = slim.learning.create_train_op(cross_entropy_loss, optimizer)
		init_op = tf.global_variables_initializer()

		final_loss = slim.learning.train(
			train_op,
			logdir='./logdir/',
			number_of_steps=2000, 
			save_summaries_secs=1, 
			log_every_n_steps=10)

		print('Finished training. Final batch loss {}'.format(final_loss))

  
		# with tf.Session() as sess:
		# 	# Run the init_op, evaluate the model outputs and print the results:
		# 	sess.run(init_op)
		# 	summary = sess.run(summary_op)
		# 	probabilities, loss, acc, confidence = sess.run([predictions, cross_entropy_loss, accuracy, mean_confidence])

		# print('summary ', summary)
		# print(probabilities)
		# print(loss)
		# print(acc)
		# print(confidence)
		
	# Forward pass 
	# TODO

	# Adding scalar summaries to the tensorboard.
	# TODO Log loss, acc, confidence (train and test)
	# for key, value in loss_dict.iteritems():
	# 	tf.summary.scalar(key, value)

	# # Adding Image summaries to the tensorboard.
	# tf.summary.image('image/0_content_inputs', content_inputs_, 3)
	# tf.summary.image('image/1_style_inputs_orig', style_inputs_orig_, 3)
	# tf.summary.image('image/2_style_inputs_aug', style_inputs_, 3)
	# tf.summary.image('image/3_stylized_images', stylized_images, 3)

	# optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
	# train_op = slim.learning.create_train_op(
	#   total_loss,
	#   optimizer,
	#   clip_gradient_norm=FLAGS.clip_gradient_norm,
	#   summarize_gradients=False)

	# # Run training
	# slim.learning.train(
	#   train_op=train_op,
	#   logdir=os.path.expanduser(FLAGS.train_dir),
	#   master=FLAGS.master,
	#   is_chief=FLAGS.task == 0,
	#   number_of_steps=FLAGS.train_steps,
	#   init_fn=init_sub_networks,
	#   save_summaries_secs=FLAGS.save_summaries_secs,
	#   save_interval_secs=FLAGS.save_interval_secs)

	# final_loss = slim.learning.train(
	#     train_op,
	#     logdir=ckpt_dir,
	#     number_of_steps=5000,
	#     save_summaries_secs=5,
	#     log_every_n_steps=500)




if __name__ == '__main__':
	# tfe.enable_eager_execution()
	# print('Loading data...')
	
	# print('Testing...')
	# evalaute_discriminator(images, labels, img_paths)
	main()
  # tf.app.run()