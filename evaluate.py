from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np

import math
import os.path
import time

import image_captioning_model

slim = tf.contrib.slim

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("eval_dir", "", "Directory to write event logs.")

def evaluate_model(sess, model, global_step, summary_writer, summary_op):

	print (" I am here")
	summary_str = sess.run(summary_op)
	print("Well done")
	summary_writer.add_summary(summary_str, global_step)
	print("Computing eval batches")
	num_eval_batches = int(math.ceil(80/8))
	print("Computed eval batches")
	print(num_eval_batches)
	start_time = time.time()
	sum_losses = 0
	sum_weights = 0.

	for i in xrange(num_eval_batches):
		print(i)
		print("computing losses...")
		cross_entropy_losses, weights = sess.run([model.target_cross_entropy_losses,model.target_cross_entropy_loss_weights])
		sum_losses += np.sum(cross_entropy_losses * weights)
		sum_weights += np.sum(weights)

		if not i % 100:
			tf.logging.info("Computed losses for %d of %d batches.", i + 1,
				num_eval_batches)
	eval_time = time.time() - start_time
	
	perplexity = math.exp(sum_losses/sum_weights)
	tf.logging.info("Perplexity = %f (%.2g sec)", perplexity, eval_time)
	
	summary = tf.Summary()
	value = summary.value.add()
	value.simple_value = perplexity
	value.tag = "Perplexity"
	summary_writer.add_summary(summary, global_step)
	#summary_writer.add_graph()
	summary_writer.flush()
	tf.logging.info("Finished processing evaluation at global step %d.",
					global_step)
	print("Finished...")
		

def run_once(model, saver, summary_writer, summary_op):
	checkpoint_dir = "/home/user/Desktop/image_captioning/model/train"
	model_path = tf.train.latest_checkpoint(checkpoint_dir)
	print("Running evaluation")
	with tf.Session() as sess:
		saver.restore(sess, model_path)
		global_step = tf.train.global_step(sess, model.global_step.name)
		print("Restored variables...")
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord = coord)
 

		try:
			print("Calling eval model")
			evaluate_model (
						    sess = sess,
						    model = model,
						    global_step = global_step,
					    	summary_writer = summary_writer,
					    	summary_op = summary_op)
		except Exception, e:
			tf.logging.error("Evaluation failed")
			coord.request_stop(e)

		coord.request_stop()	
		coord.join(threads, stop_grace_period_secs = 10)			    	

def run_eval():
	# Runs evaluation in a loop and logs summaries in TensorBoard
	input_file_pattern = 'data/mscoco/val-?????-of-00004'
	eval_dir = FLAGS.eval_dir
	if not tf.gfile.IsDirectory(eval_dir):
		tf.logging.info("Creating eval directory: %s", eval_dir)
		tf.gfile.MakeDirs(eval_dir)

	g = tf.Graph()
	with g.as_default():
		model = image_captioning_model.Image_Captioning_Model(mode = "eval", input_file_pattern = input_file_pattern) 
		model.build_graph()
		print("Model Built")	

		saver = tf.train.Saver()

		summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(eval_dir,graph=tf.get_default_graph())

		g.finalize()

		i=0
		while True:
		 	start = time.time()
		 	i = i+1
		 	print (i)
		 	tf.logging.info("Starting evaluation at " + time.strftime(
		 		"%Y-%m-%d-%H:%M:%S", time.localtime()))
			run_once(model, saver, summary_writer, summary_op)
		 	time_to_next_eval = start + 60 - time.time() # 600 is interval between logs
		 	if time_to_next_eval > 0:
		 		time.sleep(time_to_next_eval)


run_eval()






