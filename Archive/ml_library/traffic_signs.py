'''
Train a neural network to recognize traffic signs
Use German Traffic Signs Dataset for training data, and test set data
'''
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import math
import os
import time
import pickle

from config import *
from utils import *


########################################################
# Neural network architecture
########################################################
def neural_network():
    """
    Define neural network architecture
    Return relevant tensor references
    """
    with tf.variable_scope('neural_network'):
        # Tensors representing input images and labels
        x = tf.placeholder('float', [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
        y = tf.placeholder('float', [None, NUM_CLASSES])

        # Placeholder for dropout keep probability
        keep_prob = tf.placeholder(tf.float32)

        # Neural network architecture: Convolutional Neural Network (CNN)
        # Using TensorFlow-Slim to build the network:
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim

        # Use batch normalization for all convolution layers
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
            # Given x shape is (32, 32, 3)
            # Conv and pool layers
            # output shape: (32, 32, 16)
            net = slim.conv2d(x, 16, [3, 3], scope='conv0')
            # output shape: (32, 32, 16)
            net = slim.max_pool2d(
                net, [3, 3], 1, padding='SAME', scope='pool0')
            # output shape: (10, 10, 64)
            net = slim.conv2d(net, 64, [5, 5], 3,
                              padding='VALID', scope='conv1')
            # output shape: (8, 8, 64)
            net = slim.max_pool2d(net, [3, 3], 1, scope='pool1')
            # output shape: (8, 8, 128)
            net = slim.conv2d(net, 128, [3, 3], scope='conv2')
            # output shape: (8, 8, 64)
            net = slim.conv2d(net, 64, [3, 3], scope='conv3')
            # output shape: (6, 6, 64)
            net = slim.max_pool2d(net, [3, 3], 1, scope='pool3')

            # Final fully-connected layers
            net = tf.contrib.layers.flatten(net)
            net = slim.fully_connected(net, 1024, scope='fc4')
            net = tf.nn.dropout(net, keep_prob)
            net = slim.fully_connected(net, 1024, scope='fc5')
            net = tf.nn.dropout(net, keep_prob)
            net = slim.fully_connected(net, NUM_CLASSES, scope='fc6')

        # Final output (logits)
        logits = net

        # Loss (data loss and regularization loss) and optimizer
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, y))
        optimizer = OPT.minimize(loss)

        # Prediction (used during inference)
        predictions = tf.argmax(logits, 1)

        # Accuracy metric calculation
        correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # Return relevant tensor references
    return x, y, keep_prob, logits, optimizer, predictions, accuracy


########################################################
# Main training function
########################################################
def run_training():
    """
    Load training and test data
    Run training process
    Plot train/validation accuracies
    Report test accuracy
    Save model
    """
    ########################################################
    # Load training and test data
    ########################################################
    training_file = 'train_aug.p'
    testing_file = 'test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    # Basic data summary
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    image_shape = X_train.shape[1:3]
    n_classes = np.unique(y_train).shape[0]

    ########################################################
    # Data pre-processing
    ########################################################
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    ########################################################
    # Train/validation split
    ########################################################
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE)

    # Launch the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        ########################################################
        # "Instantiate" neural network, get relevant tensors
        ########################################################
        x, y, keep_prob, logits, optimizer, predictions, accuracy = neural_network()

        ########################################################
        # Training process
        ########################################################
        # TF saver to save/restore trained model
        saver = tf.train.Saver()

        if RESUME:
            print('Restoring previously trained model at %s' % MODEL_SAVE_PATH)
            # Restore previously trained model
            saver.restore(sess, MODEL_SAVE_PATH)

            # Restore previous accuracy history
            with open('accuracy_history.p', 'rb') as f:
                accuracy_history = pickle.load(f)
        else:
            print('Training model from scratch')
            # Variable initialization
            init = tf.initialize_all_variables()
            sess.run(init)

            # For book-keeping, keep track of training and validation accuracy over epochs, like such:
            # [(train_acc_epoch1, valid_acc_epoch1), (train_acc_epoch2, valid_acc_epoch2), ...]
            accuracy_history = []

        # Record time elapsed for performance check
        last_time = time.time()
        train_start_time = time.time()

        # Run NUM_EPOCH epochs of training
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

            # Print accuracy every 10 epochs
            if (epoch+1) % 10 == 0 or epoch == 0 or (epoch+1) == NUM_EPOCH:
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

        if SAVE_MODEL:
            # Save model to disk
            save_path = saver.save(sess, MODEL_SAVE_PATH)
            print('Trained model saved at: %s' % save_path)

            # Also save accuracy history
            print('Accuracy history saved at accuracy_history.p')
            with open('accuracy_history.p', 'wb') as f:
                pickle.dump(accuracy_history, f)

        # Return final test accuracy and accuracy_history
        return test_acc, accuracy_history


########################################################
# Model inference function
########################################################
def run_inference(image_files):
    """
    Load trained model and run inference on images

    Arguments:
            * images: Array of images on which to run inference

    Returns:
            * Array of strings, representing the model's predictions
    """
    # Read image files, resize them, convert to numpy arrays w/ dtype=uint8
    images = []
    for image_file in image_files:
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        image = np.array(list(image.getdata()), dtype='uint8')
        image = np.reshape(image, (32, 32, 3))

        images.append(image)
    images = np.array(images, dtype='uint8')

    # Pre-process the image (don't care about label, put dummy labels)
    images, _ = preprocess_data(images, np.array(
        [0 for _ in range(images.shape[0])]))

    with tf.Graph().as_default(), tf.Session() as sess:
        # Instantiate the CNN model
        x, y, keep_prob, logits, optimizer, predictions, accuracy = neural_network()

        # Load trained weights
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_SAVE_PATH)

        # Run inference on CNN to make predictions
        preds = sess.run(predictions, feed_dict={x: images, keep_prob: 1.})

    # Load signnames.csv to map label number to sign string
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

    final_preds = [label_map[pred] for pred in preds]

    return final_preds


if __name__ == '__main__':
    test_acc, accuracy_history = run_training()

    # Obtain list of sample image files
    sample_images = ['sample_images/' +
                     image_file for image_file in os.listdir('sample_images')]
    preds = run_inference(sample_images)
    print('Predictions on sample images:')
    for i in range(len(sample_images)):
        print('%s --> %s' % (sample_images[i], preds[i]))
