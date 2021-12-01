# Imports
import tensorflow as tf
# import cupy as cp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

from ml_library.config import *
#from ml_library.utils import *
## from ml_library.model import *
#from ml_library.w_utils import *


def random_invert_img(x, p=0.5):
    if tf.random.uniform([]) < p:
        x = (255-x)
    else:
        x
    return x


class RandomInvert(layers.Layer):
    def __init__(self, factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return random_invert_img(x)


resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    # tf.keras.layers.layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
])


# data_augmentation = keras.Sequential(
#     [
#         # resize_and_rescale,
#         # RandomInvert,
#         layers.RandomFlip("horizontal",
#                           input_shape=(IMG_SIZE,
#                                        IMG_SIZE,
#                                        NUM_CHANNELS)),
#         layers.RandomRotation(0.1),
#         layers.RandomZoom(0.1),
#     ]
# )


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding image."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """
    Maps image to a triplet (z_mean, z_log_var, z).
    """

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim, activation="gelu")
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded image vector, back into an image."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim, activation="gelu")
        self.dense_output = Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        REIN=False,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        print(original_dim)
        self.original_dim = original_dim
        print(self.original_dim)
        # self.data_aug = data_augmentation()
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        # REIN Data Augmentation
        # if(REIN):
        #     inputs = self.data_aug(inputs)
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed


def get_uncompiled_model(REIN=False):
    # https://keras.io/guides/sequential_model/ - example code to build model
    inputs = keras.Input(
        shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS), name="inputs")
    model = keras.Sequential(inputs)
    # model.add(inputs)
    # model.add(resize_and_rescale)

    # # REIN Data Augmentation
    # if(REIN):
    #     model.add(data_augmentation)

    model.add(Conv2D(16, 3, padding='same', activation='gelu'))
    # model.add(BatchNormalization)
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, strides=(3, 3), padding='same', activation='gelu'))
    # model.add(BatchNormalization)
    model.add(MaxPooling2D())
    model.add(Conv2D(128, 3, padding='same', activation='gelu'))
    # model.add(BatchNormalization)
    model.add(Conv2D(128, 3, padding='same', activation='gelu'))
    # model.add(BatchNormalization)
    model.add(MaxPooling2D())
    model.add(layers.Flatten())

    # FC Layers w/ Dropout
    model.add(Dense(1024, activation='gelu'))
    model.add(Dropout(RATE))
    model.add(Dense(1024, activation='gelu'))
    model.add(Dropout(RATE))
    outputs = model(
        Dense(NUM_CLASSES, activation="softmax", name="predictions"))
    #model.add(Dense(NUM_CLASSES, activation="softmax", name="predictions"))

    # build model
    #outputs = model(inputs)
    model2 = keras.Model(inputs=inputs, outputs=outputs)

    return model2


def get_compiled_model(REIN=False):
    model = get_uncompiled_model(REIN)

    # model.compile(
    #     optimizer="rmsprop",
    #     loss="sparse_categorical_crossentropy",
    #     metrics=["sparse_categorical_accuracy"],
    # )

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


class oldModel(tf.keras.Model):
    def __init__(self):
        super(oldModel, self).__init__()

        # Tensors representing input images and labels
        self.x = Input(
            [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], dtype=tf.dtypes.float32)
        self.y = Input([None, NUM_CLASSES], dtype=tf.dtypes.float32)

        # Placeholder for dropout keep probability
        self.rate = Input([None], dtype=tf.dtypes.float32)

        # In TF2, due to eager execution and automatic control dependencies,
        # the batch normalization moving average updates will be executed right away.
        # There is no need to separately collect them from the updates collection
        # and add them as explicit control dependencies.

        self.conv0 = Conv2D(filters=16,  kernel_size=[3, 3], activation=None)
        self.conv1 = Conv2D(filters=64,  kernel_size=[5, 5],
                            strides=3, padding='valid', activation=None)
        self.conv2 = Conv2D(filters=128, kernel_size=[3, 3], activation=None)
        self.conv3 = Conv2D(filters=64,  kernel_size=[3, 3], activation=None)

        # output shape: (32, 32, 16)
        self.pool0 = MaxPool2D(pool_size=[3, 3], strides=1, padding='same')
        self.pool1 = MaxPool2D(pool_size=[3, 3], strides=1)
        self.pool2 = MaxPool2D(pool_size=[3, 3], strides=1)

        self.fc4 = Dense(1024, activation=None)
        self.fc5 = Dense(1024, activation=None)
        self.fc6 = Dense(NUM_CLASSES, activation=None)

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
        net = Dropout(net, RATE)

        net = self.fc5(net)
        net = Dropout(net, RATE)
        net = self.fc6(net)

        # Final output (logits)
        self.logits = net
        correct_logit = tf.reduce_sum(self.y * self.logits, axis=1)
        wrong_logit = tf.reduce_max((1-self.y) * self.logits, axis=1)
        self.reg_loss = correct_logit - wrong_logit
        self.cw_grad = tf.gradients(self.reg_loss, self.x)[0]
        self.vis = tf.gradients(correct_logit, self.x)[0]

        # OPT = tf.train.GradientDescentOptimizer(learning_rate=LR)  # choose which optimizer to use
        # OPT = tf.optimizers.SGD(
        #     learning_rate=LR, momentum=0, nesterov=False, name='SGD')

        # # Loss (data loss and regularization loss) and optimizer
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     logits=self.logits, labels=self.y))
        # grad = tf.gradients(self.loss, self.x)[0]
        # self.grad_loss = tf.nn.l2_loss(grad)
        # self.optimizer = OPT.minimize(self.loss+50*self.grad_loss)

        # # Prediction (used during inference)
        # self.predictions = tf.argmax(self.logits, 1)

        # # Accuracy metric calculation
        # correct_prediction = tf.equal(self.predictions, tf.argmax(self.y, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Final output (logits)
        return self.logits
