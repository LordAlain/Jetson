from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import tensorflow.contrib.slim as slim  # TensorFlow-Slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
import math
import os
import time
import pickle

from utils import preprocess_data, next_batch, calculate_accuracy
from model import Model

model = Model()

x, y, keep_prob, logits, optimizer, predictions, accuracy = model.x, model.y, model.keep_prob, model.logits, model.optimizer, model.predictions, model.accuracy
loss = model.loss

grad = tf.sign(tf.gradients(loss, x))[0]

label_map = {}
with open('signnames.csv', 'r') as f:
    first_line = True
    for line in f:
        # Ignore first line
        if first_line:
            first_line = False
            continue

        # Populate label_map
        label_int, label_string = line.split(',')
        label_int = int(label_int)

        label_map[label_int] = label_string

with open('test.p', mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']
X_test, y_test = preprocess_data(X_test, y_test)

def attack(sess, images, labels):

	eps = 0.1
	step_size = 0.1

	gradients = sess.run(grad, feed_dict={model.x: images, model.y: labels, model.keep_prob: 1.})

	x = np.copy(images)

	x = x + step_size*gradients

	x = np.clip(x, images - eps, images + eps)
	x = np.clip(x, 0., 1.)

	return x

with tf.Session() as sess:

	saver = tf.train.Saver()
	filename = tf.train.latest_checkpoint("./model/")
	print("Latest training checkpoint is ", filename)
	if filename != None:
		saver.restore(sess, filename)
	else:
		print("No checkpoint found, exit.")
		exit()

	# Accuracy = 0
	# num = 10
	# batch_size = math.ceil(X_test.shape[0] / num)
	test_gen = next_batch(X_test, y_test, 1, True)

	for i in range(10):

		# Run testing on each batch
		images, labels = next(test_gen)

		adv_images = attack(sess, images, labels)

		# Perform gradient update (i.e. training step) on current batch
		# acc = sess.run(model.accuracy, feed_dict={x: images, y: labels, keep_prob: 1})
		# adv_acc = sess.run(model.accuracy, feed_dict={x: adv_images, y: labels, keep_prob: 1})

		idx = sess.run(model.predictions, feed_dict={x: images, y: labels, keep_prob: 1})

		adv_idx = sess.run(model.predictions, feed_dict={x: adv_images, y: labels, keep_prob: 1})

		print(label_map[idx[0]], label_map[adv_idx[0]], label_map[labels[0][0]])