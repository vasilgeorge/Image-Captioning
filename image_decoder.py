import tensorflow as tf
import numpy as np


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  		"""Prepare one image for evaluation.
  		If height and width are specified it would output an image with that size by
  		applying resize_bilinear.
  		If central_fraction is specified it would cropt the central fraction of the
  		input image.
  		Args:
    		image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      		[0, 1], otherwise it would converted to tf.float32 assuming that the range
      		is [0, MAX], where MAX is largest positive representable number for
      		int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
    		height: integer
   		 width: integer
    		central_fraction: Optional Float, fraction of the image to crop.
    		scope: Optional scope for name_scope.
		Returns:
    		3-D float Tensor of prepared image.
  		"""
		with tf.name_scope(scope, 'eval_image', [image, height, width]):
			if image.dtype != tf.float32:
				image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    		# Crop the central region of the image with an area containing 87.5% of
    		# the original image.
			if central_fraction:
				image = tf.image.central_crop(image, central_fraction=central_fraction)

			if height and width:
      		# Resize the image to the specified height and width.
				image = tf.expand_dims(image, 0)
				image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
				image = tf.squeeze(image, [0])
			image = tf.subtract(image, 0.5)
			image = tf.multiply(image, 2.0)
			return image

image = tf.read_file('./kitty.jpg')
#image_enc = tf.image.encode_jpeg(image)
image_dec = tf.image.decode_jpeg(image, channels=3)

image_prep = preprocess_for_eval(image_dec, 299, 299)
#image_fl = tf.cast(image_dec, tf.float32)
image_exp = tf.expand_dims(image_prep, 0)
#image_res = tf.image.resize_images(image_dec, [299,299])
#image_res = tf.reshape(image_dec, [1, 299, 299, 3])
#inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))
print(image_exp)

#with tf.Session() as sess:
#	#print (im_res)
#	tensor_output = sess.run(image_fl)
#	print (tensor_output)
#print (image_fl)

	
