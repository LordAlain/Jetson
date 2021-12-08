# Imports
import zipfile
import pickle
import os.path
import urllib.request
import math
from skimage import io, transform
import cv2 as cv
from tensorflow.python.ops.gen_math_ops import AccumulateNV2
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
# import ssl

from ml_library.w_utils import *
from ml_library.model import *
from ml_library.config import *

# from ml_library.utils import *


########################################################
# Helper functions and generators
########################################################


def importDatasets():
    """
    blah
    """
    TRAINING_FILE = "./Datasets/GTSRB_Final_Training_Images.zip"
    TEST_FILE = "./Datasets/GTSRB_Final_Test_Images.zip"

    # ssl._create_default_https_context = ssl._create_unverified_context
    if not os.path.exists(TRAINING_FILE):
        # Get file from URL
        url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html"
        urllib.request.urlretrieve(url, TRAINING_FILE)
        print("Downloaded Training file: ", os.path.exists(TRAINING_FILE))
    else:
        print("Training file: ", os.path.exists(TRAINING_FILE))

    if not os.path.exists(TEST_FILE):
        # Get file from URL
        url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html"
        urllib.request.urlretrieve(url, TEST_FILE)
        print("Downloaded Test file: ", os.path.exists(TEST_FILE))
    else:
        print("Test file: ", os.path.exists(TEST_FILE))


def generateTensor(archive, has_class=True):
    """
    blah
    """
    archive = zipfile.ZipFile(archive, 'r')
    file_paths = [file for file in archive.namelist()
                  if '.ppm' in file]
    tensor = {}
    tensor['features'] = []
    tensor['labels'] = []
    for filename in file_paths:
        with archive.open(filename) as img_file:

            # img = np.array(Image.open(img_file.read()))
            # img = plt.imread(img_file.read())
            img = plt.imread(img_file)
            # img = cv.cvtColor(cv.imread(img_file.read()), cv.COLOR_BGR2RGB)
            # img = io.imread(img_file.read())

            # no need to transform here, as it is done by the model
            img = transform.resize(img,
                                   output_shape=(IMG_SIZE, IMG_SIZE),
                                   mode='reflect',
                                   anti_aliasing=True
                                   )
            if has_class:
                img_class = int(filename.split('/')[-2])
            else:
                img_class = int(0)

        tensor['features'].append(img)
        tensor['labels'].append(img_class)
        # tensor['features'] = img
        # tensor['labels'] = img_class

    archive.close()
    return tensor


# def attack(sess, images, labels):
    # """
    # originally from visuliza_adv
    # """

    # eps = 0.1
    # step_size = 0.1
    # clip_range = 1

    # # eps = 3
    # # step_size = 3
    # # clip_range = 255

    # gradients = sess.run(
    #     grad, feed_dict={model.x: images, model.y: labels, model.keep_prob: 1.})

    # x = np.copy(images)

    # x = x + step_size*gradients

    # x = np.clip(x, images - eps, images + eps)
    # x = np.clip(x, 0., clip_range)

    # return x


def attack(sess, model, images, labels):
    """
    blah
    """
    # grad = tf.sign(tf.gradients(model.loss, model.x))[0]
    grad = tf.sign(tf.gradients(model.reg_loss, model.x))[0]

    # eps = 0.1
    # step_size = 0.1
    # clip_range = 1

    eps = 3
    step_size = 3
    clip_range = 255

    new_images = np.copy(images)

    for i in range(1):
        gradients = sess.run(
            grad, feed_dict={model.x: new_images, model.y: labels, model.keep_prob: 1.})
        new_images = new_images + step_size*gradients
        new_images = np.clip(new_images, images - eps, images + eps)
        new_images = np.clip(new_images, 0., clip_range)

    return new_images


def test_attack(model_file):
    """
    blah
    """
    with tf.Session() as sess:

        saver = tf.train.Saver()
        filename = tf.train.latest_checkpoint(model_file)
        print("Latest training checkpoint is ", filename)
        if filename != None:
            saver.restore(sess, filename)
        else:
            print("No checkpoint found, exit.")
            exit()

        # Accuracy = 0
        # num = 10
        # batch_size = math.ceil(x_test.shape[0] / num)
        # test_gen = next_batch(x_test, y_test, batch_size, True)

        # for i in range(num):

            # Run testing on each batch
            # images, labels = next(test_gen)

            # adv_images = attack(sess, images, labels)
            # blr_images = blur(images, 10)
            # ocl_images = ocul(images, 5)
            # drk_images = dark(images, 10)
            # drk_images = light(images, 10)

            # Perform gradient update (i.e. training step) on current batch
            # acc = sess.run(model.accuracy, feed_dict={x: images, y: labels, keep_prob: 1})
            # adv_acc = sess.run(model.accuracy, feed_dict={x: adv_images, y: labels, keep_prob: 1})
            # print("Accuracy vs Adv_Accuracy：", acc, adv_acc)
            # print("Test iter:", i, ", acc:", acc, ", adv_acc :", adv_acc)

            # grad_loss = sess.run(model.grad_loss, feed_dict={x: images, y: labels, keep_prob: 1})
            # print(grad_loss)

            # Accuracy += adv_acc
            # idx = sess.run(model.predictions, feed_dict={x: images, y: labels, keep_prob: 1})

            # adv_idx = sess.run(model.predictions, feed_dict={x: adv_images, y: labels, keep_prob: 1})
            # print("Original Label is:", label_map[np.argmax(labels[0])])
            # print(label_map[idx[0]], label_map[adv_idx[0]], label_map[labels[0][0]])
            # gradients = sess.run(model.vis, feed_dict={x: images, y: labels, keep_prob: 1})

            # plt.subplot(1, 6, 1)
            # images = images.astype(np.uint8)
            # plt.imshow(images.reshape((32, 32, 3)))
            # plt.title(label_map[idx[0]])

            # plt.subplot(1, 6, 2)
            # adv_images = adv_images.astype(np.uint8)
            # plt.imshow(adv_images.reshape((32, 32, 3)))
            # plt.title(label_map[adv_idx[0]])

            # plt.subplot(1, 6, 3)
            # blr_images = blr_images.astype(np.uint8)
            # plt.imshow(blr_images.reshape((32, 32, 3)))
            # plt.title("Gussian Blur")

            # plt.subplot(1, 6, 4)
            # drk_images = drk_images.astype(np.uint8)
            # plt.imshow(drk_images.reshape((32, 32, 3)))
            # plt.title("Lighting")

            # plt.subplot(1, 6, 5)
            # ocl_images = ocl_images.astype(np.uint8)
            # plt.imshow(ocl_images.reshape((32, 32, 3)))
            # plt.title("Occulusion")

            # plt.subplot(1, 6, 6)
            # gradients = gradients * 1000
            # gradients = np.clip(gradients, 0., 1.)*255
            # gradients = gradients.astype(np.uint8)
            # plt.imshow(gradients.reshape((32, 32, 3)))
            # plt.title("Gradients")
            # plt.show()

        # Accuracy /= num
        # print("Overall adversarial acc :", Accuracy)

        # # Show Plots
        # plt.subplot(1,2,1)
        # plt.imshow(images.reshape((32, 32, 3)))
        # plt.title(label_map[idx[0]])
        # plt.subplot(1,2,2)
        # plt.imshow(images.reshape((32, 32, 3)))
        # plt.title(label_map[adv_idx[0]])
        # plt.show()
        # print(label_map[idx[0]], label_map[adv_idx[0]])


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

    # images, labels = next(data_gen)
    # acc = sess.run(accuracy, feed_dict={x: images, y: labels, keep_prob: 1.})
    # return acc

    num_batches = math.ceil(data_size / batch_size)
    last_batch_size = data_size % batch_size

    accs = []  # accuracy for each batch

    for _ in range(num_batches):
        images, labels = next(data_gen)

        # Perform forward pass and calculate accuracy
        # Note we set keep_prob to 1.0, since we are performing inference
        acc = sess.run(accuracy, feed_dict={
                       x: images, y: labels, keep_prob: 1.})
        accs.append(acc)

    # Calculate average accuracy of all full batches (the last batch is the only partial batch)
    acc_full = np.mean(accs[:-1])

    # Calculate weighted average of accuracy accross batches
    acc = (acc_full * (data_size - last_batch_size) +
           accs[-1] * last_batch_size) / data_size

    return acc


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
        acc = sess.run(accuracy, feed_dict={
                       x: adv_images, y: labels, keep_prob: 1.})
        accs.append(acc)

    # Calculate average accuracy of all full batches (the last batch is the only partial batch)
    acc_full = np.mean(accs[:-1])

    # Calculate weighted average of accuracy accross batches
    acc = (acc_full * (data_size - last_batch_size) +
           accs[-1] * last_batch_size) / data_size

    return acc


def data_aug(orig_data):
    """
    blah
    """
    # # Load original dataset
    # with open(orig_file, mode='rb') as f:
    #     orig_data = pickle.load(f)

    orig_x, orig_y = orig_data['features'], orig_data['labels']

    # Create NUM_NEW_IMAGES new images, via image transform on random original image
    for i in range(NUM_NEW_IMAGES):
        # Pick a random image from original dataset to transform
        rand_idx = np.random.randint(orig_x.shape[0])

        # Create new image
        image = transform_image(orig_x[rand_idx], ANGLE, TRANSLATION, WARP)

        # Add new data to augmented dataset
        if i == 0:
            new_x = np.expand_dims(image, axis=0)
            new_y = np.array([orig_y[rand_idx]])
        else:
            new_x = np.concatenate((new_x, np.expand_dims(image, axis=0)))
            new_y = np.append(new_y, orig_y[rand_idx])

        if (i+1) % 1000 == 0:
            print('%d new images generated' % (i+1,))

    new_x = np.concatenate((orig_x, new_x))
    new_y = np.concatenate((orig_y, new_y))

    # # Create dict of new data
    new_data = {'features': new_x, 'labels': new_y}

    return new_data


def data_aug(orig_file, new_file):
    """
    blah
    """
    # Load original dataset
    with open(orig_file, mode='rb') as f:
        orig_data = pickle.load(f)

    orig_x, orig_y = orig_data['features'], orig_data['labels']

    # Create NUM_NEW_IMAGES new images, via image transform on random original image
    for i in range(NUM_NEW_IMAGES):
        # Pick a random image from original dataset to transform
        rand_idx = np.random.randint(orig_x.shape[0])

        # Create new image
        image = transform_image(orig_x[rand_idx], ANGLE, TRANSLATION, WARP)

        # Add new data to augmented dataset
        if i == 0:
            new_x = np.expand_dims(image, axis=0)
            new_y = np.array([orig_y[rand_idx]])
        else:
            new_x = np.concatenate((new_x, np.expand_dims(image, axis=0)))
            new_y = np.append(new_y, orig_y[rand_idx])

        if (i+1) % 1000 == 0:
            print('%d new images generated' % (i+1,))

    new_x = np.concatenate((orig_x, new_x))
    new_y = np.concatenate((orig_y, new_y))

    # Create dict of new data, and write it to disk via pickle file
    new_data = {'features': new_x, 'labels': new_y}
    with open(new_file, mode='wb') as f:
        pickle.dump(new_data, f)

    return new_data


def display_random_images(images):
    """
    Display random image, and transformed versions of it
    For debug only
    """
    image = images[np.random.randint(images.shape[0])]

    # Show original image for reference
    plt.subplot(3, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')

    for i in range(9):
        image_x = transform_image(image, ANGLE, TRANSLATION, WARP)
        plt.subplot(3, 3, i+2)
        plt.imshow(image_x)
        plt.title('Transformed Image %d' % (i+1,))

    plt.tight_layout()
    plt.show()


# def display_random(file):
    #     """
    #     Display random images from augmented dataset
    #     For debug only
    #     """
    #     with open(file, mode='rb') as f:
    #         data = pickle.load(f)
    #     images = data['features']

    #     for i in range(9):
    #         rand_idx = np.random.randint(images.shape[0])
    #         image = images[rand_idx]
    #         plt.subplot(3, 3, i+1)
    #         plt.imshow(image)
    #         plt.title('Image Idx: %d' % (rand_idx,))

    #     plt.tight_layout()
    #     plt.show()


def map_labels(label_csv):
    """
    blah
    """
    label_map = {}
    with open(label_csv, 'r') as f:
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

    return label_map


def next_batch(tensor, batch_size, augment_data):
    """
    Generator to generate data and labels
    Each batch yielded is unique, until all data is exhausted
    If all data is exhausted, the next call to this generator will throw a StopIteration

    Arguments:
        * x: image data, a tensor of shape (dataset_size, 32, 32, 3)
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

    x, y = split(tensor)

    # We know x and y are randomized from the train/validation split already,
    # so just sequentially yield the batches
    start_idx = 0
    while start_idx < x.shape[0]:
        images = x[start_idx: start_idx + batch_size]
        labels = y[start_idx: start_idx + batch_size]

        yield (np.array(images), np.array(labels))

        start_idx += batch_size


def next_batch(x, y, batch_size, augment_data):
    """
    Generator to generate data and labels
    Each batch yielded is unique, until all data is exhausted
    If all data is exhausted, the next call to this generator will throw a StopIteration

    Arguments:
        * x: image data, a tensor of shape (dataset_size, 32, 32, 3)
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

    # We know x and y are randomized from the train/validation split already,
    # so just sequentially yield the batches
    start_idx = 0
    while start_idx < x.shape[0]:
        images = x[start_idx: start_idx + batch_size]
        labels = y[start_idx: start_idx + batch_size]

        yield (np.array(images), np.array(labels))

        start_idx += batch_size


def preprocess_data(tensor):
    """
    Preprocess image data, and convert labels into one-hot

    Arguments:
        * tensor.x: Features
        * tensor.y: Labels

    Returns:
        * Preprocessed x, one-hot version of y
    """
    x, y = split(tensor)
    # Convert from RGB to grayscale if applicable
    if GRAYSCALE:
        x = rgb_to_gray(x)

    # Make all image array values fall within the range -1 to 1
    # Note all values in original images are between 0 and 255, as uint8
    x = x.astype('float32')
    x = (x - 128.) / 128.

    # Convert the labels from numerical labels to one-hot encoded labels
    y_onehot = np.zeros((y.shape[0], NUM_CLASSES))
    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]] = 1.
    y = y_onehot

    return x, y


def preprocess_data(x, y):
    """
    Preprocess image data, and convert labels into one-hot

    Arguments:
        * x: Image data
        * y: Labels

    Returns:
        * Preprocessed x, one-hot version of y
    """
    # Convert from RGB to grayscale if applicable
    if GRAYSCALE:
        x = rgb_to_gray(x)

    # Make all image array values fall within the range -1 to 1
    # Note all values in original images are between 0 and 255, as uint8
    x = x.astype('float32')
    x = (x - 128.) / 128.

    # Convert the labels from numerical labels to one-hot encoded labels
    y_onehot = np.zeros((y.shape[0], NUM_CLASSES))
    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]] = 1.
    y = y_onehot

    return x, y


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


def split(tensor):
    x, y = tensor['features'], tensor['labels']
    return x, y


def transform_image(image, angle, translation, warp):
    """
    Transform the image for data augmentation

    Arguments:
            * image: Input image
            * angle: Max rotation angle, in degrees. Direction of rotation is random.
            * translation: Max translation amount in both x and y directions,
                    expressed as fraction of total image width/height
            * warp: Max warp amount for each of the 3 reference points,
                    expressed as fraction of total image width/height

    Returns:
            * Transformed image as an np.array() object
    """
    height, width, channels = image.shape

    # Rotation
    center = (width//2, height//2)
    angle_rand = np.random.uniform(-angle, angle)
    rotation_mat = cv.getRotationMatrix2D(center, angle_rand, 1)

    image = cv.warpAffine(image, rotation_mat, (width, height))

    # Translation
    x_offset = translation * width * np.random.uniform(-1, 1)
    y_offset = translation * height * np.random.uniform(-1, 1)
    translation_mat = np.array([[1, 0, x_offset], [0, 1, y_offset]])

    image = cv.warpAffine(image, translation_mat, (width, height))

    # Warp
    # NOTE: The commented code below is left for reference
    # The warp function tends to blur the image, so it is not useds
    '''
	src_triangle = np.float32([[0, 0], [0, height], [width, 0]])
	x_offsets = [warp * width * np.random.uniform(-1, 1) for _ in range(3)]
	y_offsets = [warp * height * np.random.uniform(-1, 1) for _ in range(3)]
	dst_triangle = np.float32([[x_offsets[0], y_offsets[0]],\
							 [x_offsets[1], height + y_offsets[1]],\
							 [width + x_offsets[2], y_offsets[2]]])
	warp_mat = cv.getAffineTransform(src_triangle, dst_triangle)
	
	image = cv.warpAffine(image, warp_mat, (width, height))
	'''

    return image
