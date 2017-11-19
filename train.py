

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import image_captioning_model
tf.logging.set_verbosity(tf.logging.INFO)

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")

def train():
	batch_size = 8
	# Will be given as a parameter to the model
	input_file_pattern = '/home/user/Desktop/image_captioning/data/mscoco/train-?????-of-00256'

	# Create the training directory 
	train_dir = FLAGS.train_dir
	if not tf.gfile.IsDirectory(train_dir):
		tf.logging.info("Creating Directory: %s", train_dir)
		tf.gfile.MakeDirs(train_dir)

	

	g = tf.Graph()
	with g.as_default():

		# Create the model we are going to train
		model = image_captioning_model.Image_Captioning_Model(mode = "train", input_file_pattern = input_file_pattern) 
		model.build_graph()
		print("Model Built")

		learning_rate_decay_fn = None
		learning_rate = tf.constant(2.0)
		learning_rate_decay_factor = 0.5

		if (learning_rate_decay_factor > 0):
			num_batches_per_epoch = 586363 / batch_size # num_examples_per_epoch / batch_size
			decay_steps = int(num_batches_per_epoch * 8.0)																						

			def _learning_rate_decay_fn(learning_rate, global_step):
				return tf.train.exponential_decay(
												  learning_rate,
												  global_step,
												  decay_steps = decay_steps,
												  decay_rate  = learning_rate_decay_factor,
												  staircase = True)

			learning_rate_decay_fn = _learning_rate_decay_fn									   

		train_op = tf.contrib.layers.optimize_loss(
												   loss = model.total_loss,
												   global_step = model.global_step,
												   learning_rate = 2.0,
												   optimizer = "SGD",
												   clip_gradients = 5.0,
												   learning_rate_decay_fn = learning_rate_decay_fn)

		saver = tf.train.Saver(max_to_keep = 5)
		print(	"Starting Training")
		
		with tf.Session() as sess:
		 	
			x=slim.learning.train(
							    train_op,
							    train_dir,
							    log_every_n_steps = 1,
							    graph = g,
						    	global_step = model.global_step,
						    	number_of_steps = 1000,
						    	init_fn = model.init_fn(sess),
						    	saver = saver,
						    	save_interval_secs = 10
						   	)
			print(x)
	
train()