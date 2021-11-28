from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# # Imports
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from PIL import Image
# from sklearn.model_selection import train_test_split
# import numpy as np
# import cv2
# import math
# import os
# import time
# import pickle


# Constants
IMG_SIZE = 32  # square image of size IMG_SIZE x IMG_SIZE
GRAYSCALE = False  # convert image to grayscale?
NUM_CHANNELS = 1 if GRAYSCALE else 3
NUM_CLASSES = 43

# Image data augmentation parameters
ANGLE = 15
TRANSLATION = 0.2
WARP = 0.0  # 0.05
NUM_NEW_IMAGES = 1000
#NUM_NEW_IMAGES = 100000

# Model parameters
LR = 5e-3  # learning rate
KEEP_PROB = 0.5  # dropout keep probability
OPT = tf.optimizers.SGD(learning_rate=LR, momentum=0,
                        nesterov=False, name='SGD')
# OPT = tf.train.GradientDescentOptimizer(learning_rate=LR)  # choose which optimizer to use

# Training process
RESTORE = False  # restore previous model, don't train?
RESUME = False  # resume training from previously trained model?
NUM_EPOCH = 40
BATCH_SIZE = 128  # batch size for training (relatively small)
BATCH_SIZE_INF = 2048  # batch size for running inference, e.g. calculating accuracy
VALIDATION_SIZE = 0.2  # fraction of total training set to use as validation set
SAVE_MODEL = True  # save trained model to disk?
MODEL_SAVE_PATH = './models/model.ckpt'  # where to save trained model

TEST_SIZE = 0.2        # fraction of total training set to use as test set
VALIDATION_SIZE = 0.2  # fraction of remaining training set to use as validation set

# # Model parameters
# LR = 1e-4  # learning rate
# KEEP_PROB = 0.5  # dropout keep probability

# NUM_EPOCH = 60
# coef = 0.00005
