# Imports
import tensorflow as tf
import cupy as cp
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import (Flatten, Dense, Dropout)
import math

# Constants
IMG_SIZE = 32  # square image of size IMG_SIZE x IMG_SIZE
GRAYSCALE = False  # convert image to grayscale?
NUM_CHANNELS = 1 if GRAYSCALE else 3
NUM_CLASSES = 43

# Model parameters
LR = 5e-3  # learning rate
KEEP_PROB = 0.5  # dropout keep probability
RATE = 1 - KEEP_PROB

# Training process
RESTORE = False  # restore previous model, don't train?
RESUME = False  # resume training from previously trained model?
NUM_EPOCH = 40
BATCH_SIZE = 128  # batch size for training (relatively small)
BATCH_SIZE_INF = 2048  # batch size for running inference, e.g. calculating accuracy
VALIDATION_SIZE = 0.2  # fraction of total training set to use as validation set
SAVE_MODEL = True  # save trained model to disk?
MODEL_SAVE_PATH = './model.ckpt'  # where to save trained model

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        
        # Tensors representing input images and labels
        self.x = tf.keras.Input([None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS],dtype=tf.dtypes.float32)
        self.y = tf.keras.Input([None, NUM_CLASSES],dtype=tf.dtypes.float32)

        # Placeholder for dropout keep probability
        self.rate = tf.keras.Input([None], dtype=tf.dtypes.float32)
        
        # In TF2, due to eager execution and automatic control dependencies,
        # the batch normalization moving average updates will be executed right away.
        # There is no need to separately collect them from the updates collection 
        # and add them as explicit control dependencies.
        
        self.conv0 = Conv2D(filters = 16,  kernel_size = [3, 3], activation = None)  
        self.conv1 = Conv2D(filters = 64,  kernel_size = [5, 5],
                            strides = 3, padding ='valid', activation = None)
        self.conv2 = Conv2D(filters = 128, kernel_size = [3, 3], activation = None)
        self.conv3 = Conv2D(filters = 64,  kernel_size = [3, 3], activation = None)
        
        self.pool0 = MaxPool2D(pool_size = [3, 3], strides = 1, padding='same')  # output shape: (32, 32, 16)
        self.pool1 = MaxPool2D(pool_size = [3, 3], strides = 1)
        self.pool2 = MaxPool2D(pool_size = [3, 3], strides = 1)
        
        self.fc4 = Dense(1024, activation = None)
        self.fc5 = Dense(1024, activation = None)
        self.fc6 = Dense(NUM_CLASSES, activation = None)
        
    def call(self, inputs):
            

        # Neural network architecture: Convolutional Neural Network (CNN)
        # Given x shape is (32, 32, 3)
        # Conv and pool layers
        net = inputs
        net = self.conv0(net)  # output shape: (32, 32, 16)
        net = self.pool0(net)  # output shape: (32, 32, 16)
        net = self.conv1(net)  # output shape: (10, 10, 64)
        net = self.pool1(net)  # output shape: (8, 8, 64)
        net = self.conv2(net)  # output shape: (8, 8, 128)
        net = self.conv3(net)  # output shape: (8, 8, 64)
        net = self.pool2(net)  # output shape: (6, 6, 64)

        # Final fully-connected layers
        net = Flatten(net)
        net = self.fc4(net)
        net = Dropout(net, rate)

        net = self.fc5(net)
        net = Dropout(net, rate)
        net = self.fc6(net)
        
        # Final output (logits)
        self.logits = net
        correct_logit = tf.reduce_sum(self.y * self.logits, axis=1)
        wrong_logit = tf.reduce_max((1-self.y) * self.logits, axis=1)
        self.reg_loss =  correct_logit - wrong_logit
        self.cw_grad = tf.gradients(self.reg_loss, self.x)[0]
        self.vis = tf.gradients(correct_logit, self.x)[0]
        

        

#         OPT = tf.train.GradientDescentOptimizer(learning_rate=LR)  # choose which optimizer to use
        OPT = tf.optimizers.SGD(learning_rate=LR, momentum = 0, nesterov = False, name='SGD')
    
        # Loss (data loss and regularization loss) and optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y))
        grad = tf.gradients(self.loss, self.x)[0]
        self.grad_loss = tf.nn.l2_loss(grad)
        self.optimizer = OPT.minimize(self.loss+50*self.grad_loss)

        # Prediction (used during inference)
        self.predictions = tf.argmax(self.logits, 1)

        # Accuracy metric calculation
        correct_prediction = tf.equal(self.predictions, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        
        # Final output (logits)
        return self.logits