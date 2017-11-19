from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 

from tensorflow.contrib.slim.python.slim.nets import inception_v3
from preprocess_image import preprocess_for_eval

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("inception_checkpoint_file", "",
                       "Path to a pretrained inception_v3 model.")


class Inception_Graph(object):

	def __init__(self):
		
		# Basic Setup
		self.inception_variables = []
		self.inception_checkpoint_file = FLAGS.inception_checkpoint_file	
		self.image_decoded = None
		self.image_resized = None
		self.image_reshaped = None

	def test_inception(self):

		num_classes = 1001
		batch_size = 1

		# Read input
		self.image = tf.read_file('./images/python.jpg')
		image_decoded = tf.image.decode_jpeg(self.image, channels=3)

		image_prep = preprocess_for_eval(image_decoded, 299, 299)
		image_exp = tf.expand_dims(image_prep, 0)

		# Restore trained variables from checkpoint file
		with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
			net, end_points = inception_v3.inception_v3_base(image_exp)
		# Output : 1 x 8 x 8 x 2048
		shape = net.get_shape()
		net = slim.avg_pool2d(net, shape[1:3], padding="VALID")
		# Output : 1 x 1 x 1 x 2048
		net = slim.dropout(net)
		# Output : 1 x 8 x 8 x 2048
		net = slim.flatten(net)
		# Output : 1 x 2048

		saver = tf.train.Saver(max_to_keep = None)

		with tf.Session() as sess:		
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, self.inception_checkpoint_file)
			output = sess.run(net)			
			print (output)	
		
# Test the model
output = Inception_Graph()
output.test_inception()

																																																																																																																																	








