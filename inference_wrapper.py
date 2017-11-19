from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

import image_captioning_model



class InferenceWrapper(object):
	"""docstring for ClassName"""
	def __init__(self):
	#	super(InferenceWrapper, self).__init__()
		self.input_file_pattern = '/home/user/Desktop/image_captioning/data/mscoco/test-?????-of-00008'
		self.model = None

	def build_model(self):
		#print("1")
		model = image_captioning_model.Image_Captioning_Model(mode = "inference", input_file_pattern = self.input_file_pattern)
		model.build_graph()
		self.model = model
		return model


	def _create_restore_fn(self, checkpoint_path, saver):
		checkpoint_dir = "/home/user/Desktop/image_captioning/Pretrained-Show-and-Tell-model-master"
		checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

		def restore_fn(sess):
			#print("Restoring..")
			self.model.init_fn(sess)
			saver.restore(sess, checkpoint_path)
		return restore_fn
				
	def build_graph_from_checkpoint(self, checkpoint_path):
		self.build_model()
		saver = tf.train.Saver()

		return self._create_restore_fn(checkpoint_path, saver)

	def feed_image(self, sess, encoded_image):
		initial_state = sess.run(fetches="lstm/initial_state:0",
								 feed_dict={"image_feed:0": encoded_image})
		return initial_state

	def inference_step(self, sess, input_feed, state_feed):
		softmax_output, state_output = sess.run(
												fetches=["softmax:0", "lstm/state:0"],
												feed_dict={
															"input_feed:0": input_feed,
															"lstm/state_feed:0": state_feed,
														   }
												)
		return softmax_output, state_output, None	





