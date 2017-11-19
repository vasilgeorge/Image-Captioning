from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 

from tensorflow.contrib.slim.python.slim.nets import inception_v3
from preprocess_image import preprocess_for_eval
import inputs as inputs_ops
import image_processing

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


class Image_Captioning_Model(object):

	def __init__(self, mode, input_file_pattern):
		
		# Basic Setup
		self.input_file_pattern = input_file_pattern
		self.mode = mode
		self.inception_variables = []
		self.inception_checkpoint_file = './inception_v3.ckpt'	
		self.image_decoded = None
		self.image_resized = None
		self.image_reshaped = None
		self.images1 = None
		self.input_seqs = None
		self.input_mask = None
		self.target_seqs = None
		self.image_emb = None
		self.initializer = tf.random_uniform_initializer(minval = -0.08, maxval = 0.08)	
		self.reader = tf.TFRecordReader()
		self.global_step = None
		self.feat_images = None
		self.total_loss = None
		self.target_cross_entropy_losses = None
		self.target_cross_entropy_loss_weights = None
		self.init_fn = None
		self.inception_variables=[]
		self.batch_size = 8

	def read_inputs(self):

		
		if self.mode == 'inference':
			image_feed = tf.placeholder(dtype = tf.string, shape = [], name = "image_feed")

			input_feed = tf.placeholder(dtype = tf.int64, shape = [None], name = "input_feed")

			images = tf.expand_dims(image_processing.process_image(image_feed, is_training = False, 
																   height = 299, width = 299, thread_id = 0, 
																   image_format = 'jpeg'), 0)
			input_seqs = tf.expand_dims(input_feed, 1)
			target_seqs = None
			input_mask = None
		else:
			input_queue = inputs_ops.prefetch_input_data(self.reader,
														file_pattern = self.input_file_pattern, # Must be given as input
														is_training = True,
														batch_size = 32, 
														values_per_shard = 2300,
														input_queue_capacity_factor = 2,
														num_reader_threads = 1)
			images_and_captions = []
			for thread_id in range(4):
				serialized_sequence_example = input_queue.dequeue()
				encoded_image, caption = inputs_ops.parse_sequence_example(serialized_sequence_example,
																			image_feature="image/data",
																			caption_feature="image/caption_ids")
				image = image_processing.process_image(encoded_image, is_training = False, height = 299, width = 299, 
													   thread_id=thread_id, image_format="jpeg")
				images_and_captions.append([image, caption])

			queue_capacity = (2 * 4 * self.batch_size)
			images, input_seqs, target_seqs, input_mask = (
												inputs_ops.batch_with_dynamic_pad(images_and_captions,
												batch_size=32,
												queue_capacity=queue_capacity))

		self.images = images
 		self.input_seqs = input_seqs
		self.target_seqs = target_seqs
		self.input_mask = input_mask

			

	def inception_graph(self):
		# Builds the graph for inception_v3, loads the graph variables
		# (weights, biases, etc.) from the checkpoint and returns the 
		# feature map of the input image. - Need to set the variables of
		# inception graph to trainable = False.
		#if (self.mode!='inference'):
		
		# Restore trained variables from checkpoint file
		with slim.arg_scope([slim.conv2d,slim.batch_norm], trainable = False):	
			with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
				net, end_points = inception_v3.inception_v3_base(self.images)
		self.inception_variables = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")		

		# Output : batch_size x 8 x 8 x 2048
		shape = net.get_shape()																																	
		net = slim.avg_pool2d(net, shape[1:3], padding="VALID")
		# Output : batch_size x 1 x 1 x 2048
		net = slim.dropout(net)
		# Output : batch_size x 8 x 8 x 2048
		feat_images = slim.flatten(net)
		# Output : batch_size x 2048 - Input to image_embedder
		
		
																																																																																																																																																																																																																																																																																																																																																			
		self.feat_images = feat_images																							

	def image_embedder(self,dimension):
		# Embeds the 2048-dimensional feature map of the input image
		# to a 512-dimensional embedding vector. 
		# The weights of this layer need to be trained.
		with tf.variable_scope("image_embedding") as scope:
			image_embeddings = tf.contrib.layers.fully_connected(inputs = self.feat_images,
																num_outputs = dimension,
																activation_fn = None,
																weights_initializer = tf.random_uniform_initializer(minval = -0.08,
																													maxval = 0.08),
																biases_initializer = None,
																scope = scope)
		tf.constant(512, name="embedding_size")	
		self.image_emb = image_embeddings
			
	def word_embeddings(self):
		# Embeds the words into word embedding vectors.
		
		with tf.variable_scope("seq_embedding") as scope:
			# Creates the embedding map of vacbulary_size * embedding_dimensions.
			embedding_map = tf.get_variable( name = "map",
											 shape = [12000, 512],
											 initializer = self.initializer)

			# Returns the rows of the embedding map that correspond to the input sequences [batch_size, padded_length, 512].
			word_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)
			#word_embeddings = tf.expand_dims(word_embeddings,0) # Input to LSTM must be of 3 dimensions.

		self.word_embeddings = word_embeddings
		#print(self.word_embeddings)

	def LSTM_Model(self):
		
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = 512, state_is_tuple = True)

		if self.mode == 'train':
			lstm_cell = tf.contrib.rnn.DropoutWrapper(
													 lstm_cell,
													 input_keep_prob = 0.7,
													 output_keep_prob = 0.7)
		

		with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
			# Run a single LSTM step given as input the image embedding.
			zero_state = lstm_cell.zero_state(batch_size = self.image_emb.get_shape()[0],
											  dtype = tf.float32)
			#print(zero_state)
			_, initial_state = lstm_cell(self.image_emb, zero_state)

			
			lstm_scope.reuse_variables()

			if self.mode == 'inference':

				tf.concat(axis = 1, values = initial_state, name = 'initial_state')

				state_feed = tf.placeholder(dtype = tf.float32,
											shape = [None, sum(lstm_cell.state_size)],
											name = 'state_feed')
				state_tuple = tf.split(value = state_feed, num_or_size_splits = 2, axis = 1)

				lstm_out, state_tuple = lstm_cell (inputs = tf.squeeze(self.word_embeddings, axis = [1]),
												   state = state_tuple)
				#print(state_tuple)
				tf.concat(axis = 1, values = state_tuple, name = 'state')

			else:
				# Unroll the RNN - Output dimensions : [batch_size, padded_length, 512] 
				sequence_length = tf.reduce_sum(self.input_mask, 1)
				lstm_out, _ = tf.nn.dynamic_rnn(cell = lstm_cell,
										 inputs = self.word_embeddings,
										 sequence_length = sequence_length,
										 initial_state = initial_state,
										 dtype = tf.float32,
										 scope = lstm_scope)	

			
			# Stack batches vertically - Output dimensions : [batch_size * padded_length, 512]
			lstm_out = tf.reshape(lstm_out, [-1, lstm_cell.output_size])
			#print(lstm_out)

		# Use of a fully convolutional layer to depict the output of the LSTM(512) to
		# the vocabulary size(12000) to get the logits corresponding to the tokenized words.
		# Output dimensions : [batch_size * padded_length, 12000] - Needs to be trained.
		with tf.variable_scope("logits") as logits_scope:
			logits = tf.contrib.layers.fully_connected(inputs = lstm_out,
													   num_outputs = 12000,
													   activation_fn = None,
													   weights_initializer = self.initializer,
													   scope = logits_scope,
													   reuse = False
													   )


			#print(logits)
		if self.mode == 'inference':
			tf.nn.softmax(logits, name = "softmax")
		else:	
			#print( "Rnn Working...")	
			targets = tf.reshape(self.target_seqs, [-1])
			weights = tf.to_float(tf.reshape(self.input_mask, [-1]))
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets,
																  logits = logits)
			batch_loss = tf.div(tf.reduce_sum(tf.multiply(loss,weights)),
								tf.reduce_sum(weights),
								name = "batch_loss")
			tf.losses.add_loss(batch_loss)
			total_loss = tf.losses.get_total_loss()

			#for var in tf.trainable_variables():
   				# print (var.name)
			# Add summaries..
			tf.summary.scalar("losses/batch_loss", batch_loss)
			tf.summary.scalar("losses/total_loss", total_loss)
			for var in tf.trainable_variables():
				tf.summary.histogram("parameters/" + var.op.name, var)

			# Should add the graph so that Tensorboard will be able to depict the graph.
			self.total_loss = total_loss
			self.target_cross_entropy_losses = loss
			self.target_cross_entropy_loss_weights = weights 
	
	def restore_inception(self):

		saver = tf.train.Saver(self.inception_variables)
		def restore_fn(sess):
			#print("5")
			#with tf.Session() as sess:		
			#sess.run(tf.global_variables_initializer())
			saver.restore(sess, self.inception_checkpoint_file)
				#output = sess.run(feat_image)			
			#sess.close()
		self.init_fn = restore_fn									
				

	def setup_global_step(self):
		# Setup Global Step
		global_step  = tf.Variable(
								  initial_value = 0,
								  name = "global_step",
								  trainable = False,
								  collections = [tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

		self.global_step = global_step


	def build_graph(self):
		graph_inputs = self.read_inputs()
		#print("2")
		feat_images = self.inception_graph()
		image_emb = self.image_embedder(512) # 1 * 512
		word_emb = self.word_embeddings() # Input needs to be defined # ? * 512
		lstm_graph = self.LSTM_Model()
		self.restore_inception()
		global_step = self.setup_global_step()


# Test the model
# model = Image_Captioning_Model(mode="train", input_file_pattern = 'data/mscoco/train-?????-of-00256') # Should give an input image as argument
# model.build_graph()
# with tf.Session() as sess:
# 	fn = model.init_fn(sess)

																																																																																																																																	








