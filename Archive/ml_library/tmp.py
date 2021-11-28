from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import math
import os
import time
import pickle


from utils import next_batch, calculate_accuracy, calculate_adv_accuracy, attack
from model import Model

def preprocess_data(X, y):
    # Make all image array values fall within the range -1 to 1
    # Note all values in original images are between 0 and 255, as uint8
    X = X.astype('float32')
#     X = (X - 128.) #/ 128.

    # Convert the labels from numerical labels to one-hot encoded labels
    y_onehot = np.zeros((y.shape[0], NUM_CLASSES))
    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]] = 1.
    y = y_onehot

    return X, y

# Settings/parameters to be used later

# Constants
IMG_SIZE = 32  # square image of size IMG_SIZE x IMG_SIZE
GRAYSCALE = False  # convert image to grayscale?
NUM_CHANNELS = 1 if GRAYSCALE else 3
NUM_CLASSES = 43

# Model parameters
LR = 1e-3  # learning rate
KEEP_PROB = 0.5  # dropout keep probability
# OPT = tf.train.GradientDescentOptimizer(learning_rate=LR)  # choose which optimizer to use
OPT = tf.optimizers.SGD (learning_rate=LR,
                         lr_decay=0.0,
                         decay_step=100,
                         staircase=False,
                         use_locking=False,
                         name='SGD'
                        )
# Training process
RESTORE = False  # restore previous model, don't train?
RESUME = False  # resume training from previously trained model?

NUM_EPOCH = 100
coef = 1
model_dir = "./natural_model/"

BATCH_SIZE = 128  # batch size for training (relatively small)
BATCH_SIZE_INF = 2048  # batch size for running inference, e.g. calculating accuracy
VALIDATION_SIZE = 0.2  # fraction of total training set to use as validation set
SAVE_MODEL = True  # save trained model to disk?

# Load pickled data

with open('train_aug.p', mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_train, y_train = preprocess_data(X_train, y_train)


with open('test.p', mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']
X_test, y_test = preprocess_data(X_test, y_test)


#Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE)

model = Model()

x, y, logits, predictions, accuracy = model.x, model.y, model.logits, model.predictions, model.accuracy
keep_prob = model.keep_prob
loss = model.loss


global_step = tf.train.get_or_create_global_step()

grad_loss =  tf.reduce_sum(tf.nn.l2_loss(model.cw_grad))

total_loss = model.loss #+ coef * grad_loss

OPT = tf.train.GradientDescentOptimizer(learning_rate=LR) 

optimizer = OPT.minimize(total_loss, global_step=global_step)

train_acc = tf.summary.scalar("Train accuracy", accuracy)
test_acc = tf.summary.scalar("Test accuracy", accuracy)
adv_acc = tf.summary.scalar("Adv accuracy", accuracy)
grad_loss = tf.summary.scalar("gradients l2 loss", grad_loss)
cros_loss = tf.summary.scalar('cross entropy', loss)

# sess = tf.Session()
# if True:
with tf.Session() as sess:

	saver = tf.train.Saver()
	filename = tf.train.latest_checkpoint(model_dir)
	print("Latest training checkpoint is ", filename)
	if filename != None:
		saver.restore(sess, filename)
	else:
		sess.run(tf.global_variables_initializer())

	summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

	last_time = time.time()
	train_start_time = time.time()
	accuracy_history = []

	for epoch in range(NUM_EPOCH):
		# Instantiate generator for training data
		train_gen = next_batch(X_train, y_train, BATCH_SIZE, True)

		# How many batches to run per epoch
		num_batches_train = math.ceil(X_train.shape[0] / BATCH_SIZE)

		# Run training on each batch
		for _ in range(num_batches_train):
			# Obtain the training data and labels from generator
			images, labels = next(train_gen)

			# Perform gradient update (i.e. training step) on current batch
			sess.run(optimizer, feed_dict={x: images, y: labels, keep_prob: KEEP_PROB})

		train_gen = next_batch(X_train, y_train, 2000, True)
		images, labels = next(train_gen)
		strain_acc, sgrad_loss, scros_loss = sess.run([train_acc, grad_loss, cros_loss], feed_dict={x: images, y: labels, keep_prob: KEEP_PROB})
		summary_writer.add_summary(strain_acc, global_step.eval(sess))
		summary_writer.add_summary(sgrad_loss, global_step.eval(sess))
		summary_writer.add_summary(scros_loss, global_step.eval(sess))

		test_gen = next_batch(X_test, y_test, X_test.shape[0], True)
		images, labels = next(test_gen)
		test_summary = sess.run(test_acc, feed_dict={x: images, y: labels, keep_prob: 1.})
		summary_writer.add_summary(test_summary, global_step.eval(sess))

		test_gen = next_batch(X_test, y_test, 2000, True)
		images, labels = next(test_gen)
		adv_images = attack(sess, model, images, labels)
		adv_test_summary = sess.run(adv_acc, feed_dict={x: adv_images, y: labels, keep_prob: 1.})
		summary_writer.add_summary(adv_test_summary, global_step.eval(sess))

		# Calculate training and validation accuracy across the *entire* train/validation set
		# If train/validation size % batch size != 0
		# then we must calculate weighted average of the accuracy of the final (partial) batch,
		# w.r.t. the rest of the full batches

		# # Training set
		# train_gen = next_batch(X_train, y_train, BATCH_SIZE_INF, True)
		# train_size = X_train.shape[0]
		# train_acc = calculate_accuracy(train_gen, train_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)
        
		test_gen = next_batch(X_test, y_test, X_test.shape[0], True)
		images, labels = next(test_gen)
		np_test_acc = sess.run(accuracy, feed_dict={x: images, y: labels, keep_prob: 1.})
		# test_size = X_test.shape[0]
		# np_test_acc = calculate_accuracy(test_gen, test_size, test_size, accuracy, x, y, keep_prob, sess)

		# adtest_gen = next_batch(X_test, y_test, BATCH_SIZE_INF, False)
		# adtest_size = X_test.shape[0]
		# adtest_acc = calculate_adv_accuracy(model, adtest_gen, adtest_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)

		# # Record and report train/validation/test accuracies for this epoch
		# accuracy_history.append((train_acc, valid_acc))

		print('Epoch %d -- Test acc.: %.4f, Elapsed time: %.2f sec' %\
		    (epoch+1, np_test_acc, time.time() - last_time))
		last_time = time.time()

		if epoch % 10 == 9:
		#     # Also save accuracy history
		#     print('Accuracy history saved at accuracy_history.p')
		#     with open('accuracy_history.p', 'wb') as f:
		#         pickle.dump(accuracy_history, f)
		    saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
		    print('Model Saved !!!', epoch, "\n")

	total_time = time.time() - train_start_time
	print('Total elapsed time: %.2f sec (%.2f min)' % (total_time, total_time/60))

	# After training is complete, evaluate accuracy on test set
	print('Calculating test accuracy...')
	test_gen = next_batch(X_test, y_test, BATCH_SIZE_INF, False)
	test_size = X_test.shape[0]
	test_acc = calculate_accuracy(test_gen, test_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)
	print('Test acc.: %.4f' % (test_acc,))