from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf 

import image_captioning_model
import inference_wrapper
import vocabulary
import caption_generator


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

def run_inference():

	input_file_pattern = '/home/user/Desktop/image_captioning/data/mscoco/test-?????-of-00008'

	g = tf.Graph()
	with g.as_default():
		model = inference_wrapper.InferenceWrapper()
		restore_fn = model.build_graph_from_checkpoint('home/user/Desktop/image_captioning/Pretrained-Show-and-Tell-model-master')

	g.finalize()
	
	# Create a vocabulary

	vocab = vocabulary.Vocabulary('./home/user/Desktop/image_captioning/Pretrained-Show-and-Tell-model-master/word_counts.txt')

	filenames = []
	for file_pattern in FLAGS.input_files.split(","):	
		filenames.extend(tf.gfile.Glob(file_pattern))	

	with tf.Session(graph = g) as sess:

		restore_fn(sess)

		generator = caption_generator.CaptionGenerator(model, vocab)


		for filename in filenames:
			with tf.gfile.GFile(filename, 'r') as f:
				image = f.read()
			captions = generator.beam_search(sess, image)
			
			print("Captions for image %s:" % os.path.basename(filename))		
			for i, caption in enumerate(captions):
				sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
				sentence = " ".join(sentence)
				print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

run_inference()