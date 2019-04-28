from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

def get_files(filename,channel):
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
									   features={
										   'label': tf.FixedLenFeature([], tf.int64),
										   'img_raw1': tf.FixedLenFeature([], tf.string),
										   'img_raw2': tf.FixedLenFeature([], tf.string),
									   })
	img1 = tf.decode_raw(features['img_raw1'], tf.uint8)
	img2 = tf.decode_raw(features['img_raw2'], tf.uint8)

	img1 = tf.reshape(img1, [10,100,100, channel])
	img2 = tf.reshape(img2, [10,100,100, channel])
	image1 = tf.cast(img1, tf.float32) * (1. / 255)
	image2 = tf.cast(img2, tf.float32) * (1. / 255)
	label = tf.cast(features['label'], tf.int32)

	return image1,image2, label



def inputs(filename_queue,batch_size,channel):

	#filename_queue = tf.train.string_input_producer([filename],num_epochs=num_epochs)
	image1, image2,  label = get_files(filename_queue,channel)
	images1, images2, labels = tf.train.shuffle_batch([image1,image2, label],batch_size=batch_size, num_threads=4 , capacity=300,min_after_dequeue=100)

	labels = tf.one_hot(labels, 2)
	return images1,images2, labels

