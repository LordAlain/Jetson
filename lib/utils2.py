from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import cv2 as cv
import math
import os
import time
import pickle

print(tf.__version__)
# Settings/parameters to be used later

# Constants
IMG_SIZE = 32  # square image of size IMG_SIZE x IMG_SIZE
GRAYSCALE = False  # convert image to grayscale?
NUM_CHANNELS = 1 if GRAYSCALE else 3
NUM_CLASSES = 43

# Model parameters
LR = 5e-3  # learning rate
KEEP_PROB = 0.5  # dropout keep probability
OPT = tf.optimizers.SGD (learning_rate=LR, momentum = 0, nesterov = False, name='SGD')
# OPT = tf.train.GradientDescentOptimizer(learning_rate=LR)  # choose which optimizer to use

# Training process
RESTORE = False  # restore previous model, don't train?
RESUME = False  # resume training from previously trained model?
NUM_EPOCH = 40
BATCH_SIZE = 128  # batch size for training (relatively small)
BATCH_SIZE_INF = 2048  # batch size for running inference, e.g. calculating accuracy
VALIDATION_SIZE = 0.2  # fraction of total training set to use as validation set
SAVE_MODEL = True  # save trained model to disk?
MODEL_SAVE_PATH = './model.ckpt'  # where to save trained model

########################################################
# Helper functions and generators
########################################################
def rgb_to_gray(images):
    """
    Convert batch of RGB images to grayscale
    Use simple average of R, G, B values, not weighted average

    Arguments:
        * Batch of RGB images, tensor of shape (batch_size, 32, 32, 3)

    Returns:
        * Batch of grayscale images, tensor of shape (batch_size, 32, 32, 1)
    """
    images_gray = np.average(images, axis=3)
    images_gray = np.expand_dims(images_gray, axis=3)
    return images_gray


def preprocess_data(X, y):
    """
    Preprocess image data, and convert labels into one-hot

    Arguments:
        * X: Image data
        * y: Labels

    Returns:
        * Preprocessed X, one-hot version of y
    """
    # Convert from RGB to grayscale if applicable
    if GRAYSCALE:
        X = rgb_to_gray(X)

    # Make all image array values fall within the range -1 to 1
    # Note all values in original images are between 0 and 255, as uint8
    X = X.astype('float32')
    X = (X - 128.) / 128.

    # Convert the labels from numerical labels to one-hot encoded labels
    y_onehot = np.zeros((y.shape[0], NUM_CLASSES))
    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]] = 1.
    y = y_onehot

    return X, y


def next_batch(X, y, batch_size, augment_data):
    """
    Generator to generate data and labels
    Each batch yielded is unique, until all data is exhausted
    If all data is exhausted, the next call to this generator will throw a StopIteration

    Arguments:
        * X: image data, a tensor of shape (dataset_size, 32, 32, 3)
        * y: labels, a tensor of shape (dataset_size,)  <-- i.e. a list
        * batch_size: Size of the batch to yield
        * augment_data: Boolean value, whether to augment the data (i.e. perform image transform)

    Yields:
        A tuple of (images, labels), where:
            * images is a tensor of shape (batch_size, 32, 32, 3)
            * labels is a tensor of shape (batch_size,)
    """
    # A generator in this case is likely overkill,
    # but using a generator is a more scalable practice,
    # since future datasets may be too large to fit in memory

    # We know X and y are randomized from the train/validation split already,
    # so just sequentially yield the batches
    start_idx = 0
    while start_idx < X.shape[0]:
        images = X[start_idx : start_idx + batch_size]
        labels = y[start_idx : start_idx + batch_size]

        yield (np.array(images), np.array(labels))

        start_idx += batch_size


def calculate_accuracy(data_gen, data_size, batch_size, accuracy, x, y, keep_prob, sess):
    """
    Helper function to calculate accuracy on a particular dataset

    Arguments:
        * data_gen: Generator to generate batches of data
        * data_size: Total size of the data set, must be consistent with generator
        * batch_size: Batch size, must be consistent with generator
        * accuracy, x, y, keep_prob: Tensor objects in the neural network
        * sess: TensorFlow session object containing the neural network graph

    Returns:
        * Float representing accuracy on the data set
    """

    images, labels = next(data_gen)
    acc = sess.run(accuracy, feed_dict={x: images, y: labels, keep_prob: 1.})
    return acc

    # num_batches = math.ceil(data_size / batch_size)
    # last_batch_size = data_size % batch_size

    # accs = []  # accuracy for each batch

    # for _ in range(num_batches):
    #     images, labels = next(data_gen)

    #     # Perform forward pass and calculate accuracy
    #     # Note we set keep_prob to 1.0, since we are performing inference
    #     acc = sess.run(accuracy, feed_dict={x: images, y: labels, keep_prob: 1.})
    #     accs.append(acc)

    # # Calculate average accuracy of all full batches (the last batch is the only partial batch)
    # acc_full = np.mean(accs[:-1])

    # # Calculate weighted average of accuracy accross batches
    # acc = (acc_full * (data_size - last_batch_size) + accs[-1] * last_batch_size) / data_size

    # return acc

def attack(sess, model, images, labels):

    loss = model.loss

    grad = tf.sign(tf.gradients(loss, model.x))[0]

    eps = 3
    step_size = 3
    x = np.copy(images)
    
    for i in range(1):
        gradients = sess.run(grad, feed_dict={model.x: x, model.y: labels, model.keep_prob: 1.})
        x = x + step_size*gradients
        x = np.clip(x, images - eps, images + eps)
        x = np.clip(x, 0., 255.)

    return x

def calculate_adv_accuracy(model, data_gen, data_size, batch_size, accuracy, x, y, keep_prob, sess):
    """
    Helper function to calculate accuracy on a particular dataset

    Arguments:
        * data_gen: Generator to generate batches of data
        * data_size: Total size of the data set, must be consistent with generator
        * batch_size: Batch size, must be consistent with generator
        * accuracy, x, y, keep_prob: Tensor objects in the neural network
        * sess: TensorFlow session object containing the neural network graph

    Returns:
        * Float representing accuracy on the data set
    """
    num_batches = math.ceil(data_size / batch_size)
    last_batch_size = data_size % batch_size

    accs = []  # accuracy for each batch

    for _ in range(num_batches):
        images, labels = next(data_gen)

        adv_images = attack(sess, model, images, labels)
        # Perform forward pass and calculate accuracy
        # Note we set keep_prob to 1.0, since we are performing inference
        acc = sess.run(accuracy, feed_dict={x: adv_images, y: labels, keep_prob: 1.})
        accs.append(acc)

    # Calculate average accuracy of all full batches (the last batch is the only partial batch)
    acc_full = np.mean(accs[:-1])

    # Calculate weighted average of accuracy accross batches
    acc = (acc_full * (data_size - last_batch_size) + accs[-1] * last_batch_size) / data_size

    return acc