from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import numpy as np
import math
import zipfile
# import pickle

from config import *
from utils import *
from model import *

model = Model()

x, y, keep_prob, logits, optimizer, predictions, accuracy = model.x, model.y, model.keep_prob, model.logits, model.optimizer, model.predictions, model.accuracy
loss = model.loss

grad = tf.sign(tf.gradients(loss, x))[0]

label_map = map_labels('signnames.csv')

testing_file = './Datasets/GTSRB_Final_Test_Images.zip'
test = generateTensor(testing_file)
x_test, y_test = preprocess_data(test)


def attack(sess, images, labels):

    eps = 0.1
    step_size = 0.1

    gradients = sess.run(
        grad, feed_dict={model.x: images, model.y: labels, model.keep_prob: 1.})

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

    Accuracy = 0
    num = 10
    batch_size = math.ceil(x_test.shape[0] / num)
    test_gen = next_batch(x_test, y_test, batch_size, True)

    for i in range(num):

        # Run testing on each batch
        images, labels = next(test_gen)

        adv_images = attack(sess, images, labels)

        # Perform gradient update (i.e. training step) on current batch
        acc = sess.run(model.accuracy, feed_dict={
                       x: images, y: labels, keep_prob: 1})
        adv_acc = sess.run(model.accuracy, feed_dict={
                           x: adv_images, y: labels, keep_prob: 1})

        print("Test iter", i, "acc :", acc, "adv_acc :", adv_acc)

        Accuracy += adv_acc
        # idx = sess.run(model.predictions, feed_dict={x: images, y: labels, keep_prob: 1})
        # print(label_map[idx[0]])

    Accuracy /= num
    print("Overall adversarial acc :", Accuracy)
