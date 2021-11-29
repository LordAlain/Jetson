# Imports
import tensorflow as tf
# from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import math
# import os
import time
import pickle


from config import *
from utils import *
from model import Model


# Load pickled data

with open('train_aug.p', mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_train, y_train = preprocess_data(X_train, y_train)


with open('test.p', mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']
X_test, y_test = preprocess_data(X_test, y_test)


# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=VALIDATION_SIZE)

model = Model()

x, y, logits, predictions, accuracy = model.x, model.y, model.logits, model.predictions, model.accuracy
keep_prob = model.keep_prob

global_step = tf.train.get_or_create_global_step()

coef = 0.01

grad_loss = coef * tf.reduce_sum(tf.nn.l2_loss(model.cw_grad))

total_loss = model.loss + grad_loss

OPT = tf.train.GradientDescentOptimizer(learning_rate=LR)

optimizer = OPT.minimize(total_loss, global_step=global_step)


sess = tf.Session()
if True:

    saver = tf.train.Saver()
    filename = tf.train.latest_checkpoint("./Alexnet/")
    print("Latest training checkpoint is ", filename)
    if filename != None:
        saver.restore(sess, filename)
    else:
        sess.run(tf.global_variables_initializer())

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
            sess.run(optimizer, feed_dict={
                     x: images, y: labels, keep_prob: KEEP_PROB})
        # Calculate training and validation accuracy across the *entire* train/validation set
        # If train/validation size % batch size != 0
        # then we must calculate weighted average of the accuracy of the final (partial) batch,
        # w.r.t. the rest of the full batches

        # Training set
        train_gen = next_batch(X_train, y_train, BATCH_SIZE_INF, True)
        train_size = X_train.shape[0]
        train_acc = calculate_accuracy(
            train_gen, train_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)

        # Validation set
        valid_gen = next_batch(X_valid, y_valid, BATCH_SIZE_INF, True)
        valid_size = X_valid.shape[0]
        valid_acc = calculate_accuracy(
            valid_gen, valid_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)

        # Record and report train/validation/test accuracies for this epoch
        accuracy_history.append((train_acc, valid_acc))

        print('Epoch %d -- Train acc.: %.4f, Validation acc.: %.4f, Elapsed time: %.2f sec' %
              (epoch+1, train_acc, valid_acc, time.time() - last_time))
        last_time = time.time()

    total_time = time.time() - train_start_time
    print('Total elapsed time: %.2f sec (%.2f min)' %
          (total_time, total_time/60))

    # After training is complete, evaluate accuracy on test set
    print('Calculating test accuracy...')
    test_gen = next_batch(X_test, y_test, BATCH_SIZE_INF, False)
    test_size = X_test.shape[0]
    test_acc = calculate_accuracy(
        test_gen, test_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)
    print('Test acc.: %.4f' % (test_acc,))
