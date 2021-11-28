# Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.transform import resize
import math
import urllib.request
import os.path

from config import *

########################################################
# Helper functions and generators
########################################################


def importDatasets():
    """
    Display random image, and transformed versions of it
    For debug only
    """
    if not os.path.exists("./Datasets/GTSRB_Final_Training_Images.zip"):
        # Get file from URL
        url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html"
        filename = "./Datasets/GTSRB_Final_Training_Images.zip"
        urllib.request.urlretrieve(url, filename)

    if not os.path.exists("./Datasets/GTSRB_Final_Testing_Images.zip"):
        # Get file from URL
        url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html"
        filename = "./Datasets/GTSRB_Final_Testing_Images.zip"
        urllib.request.urlretrieve(url, filename)


def generateTensor(archive):
    file_paths = [file for file in archive.namelist()
                  if '.ppm' in file]
    tensor = {}
    for filename in file_paths:
        with archive.open(filename) as img_file:

            # img = Image.open(img_file.read())
            # img = imread(img_file.read())
            img = cv.imread(img_file.read())
            img = resize(img,
                         output_shape=(IMG_SIZE, IMG_SIZE),
                         mode='reflect',
                         anti_aliasing=True
                         )

            img_class = int(filename.split('/')[-2])

        tensor['features'].append(img)
        tensor['labels'].append(img_class)

    archive.close()
    return tensor


def attack(sess, model, images, labels):

    loss = model.loss

    grad = tf.sign(tf.gradients(loss, model.x))[0]

    eps = 3
    step_size = 3
    x = np.copy(images)

    for i in range(1):
        gradients = sess.run(
            grad, feed_dict={model.x: x, model.y: labels, model.keep_prob: 1.})
        x = x + step_size*gradients
        x = np.clip(x, images - eps, images + eps)
        x = np.clip(x, 0., 255.)

    return x


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

    orig_X, orig_y = orig_data['features'], orig_data['labels']

    # Create NUM_NEW_IMAGES new images, via image transform on random original image
    for i in range(NUM_NEW_IMAGES):
        # Pick a random image from original dataset to transform
        rand_idx = np.random.randint(orig_X.shape[0])

        # Create new image
        image = transform_image(orig_X[rand_idx], ANGLE, TRANSLATION, WARP)

        # Add new data to augmented dataset
        if i == 0:
            new_X = np.expand_dims(image, axis=0)
            new_y = np.array([orig_y[rand_idx]])
        else:
            new_X = np.concatenate((new_X, np.expand_dims(image, axis=0)))
            new_y = np.append(new_y, orig_y[rand_idx])

        if (i+1) % 1000 == 0:
            print('%d new images generated' % (i+1,))

    new_X = np.concatenate((orig_X, new_X))
    new_y = np.concatenate((orig_y, new_y))

    # # Create dict of new data
    new_data = {'features': new_X, 'labels': new_y}

    return new_data


def data_aug(orig_file, new_file):
    """
    blah
    """
    # Load original dataset
    with open(orig_file, mode='rb') as f:
        orig_data = pickle.load(f)

    orig_X, orig_y = orig_data['features'], orig_data['labels']

    # Create NUM_NEW_IMAGES new images, via image transform on random original image
    for i in range(NUM_NEW_IMAGES):
        # Pick a random image from original dataset to transform
        rand_idx = np.random.randint(orig_X.shape[0])

        # Create new image
        image = transform_image(orig_X[rand_idx], ANGLE, TRANSLATION, WARP)

        # Add new data to augmented dataset
        if i == 0:
            new_X = np.expand_dims(image, axis=0)
            new_y = np.array([orig_y[rand_idx]])
        else:
            new_X = np.concatenate((new_X, np.expand_dims(image, axis=0)))
            new_y = np.append(new_y, orig_y[rand_idx])

        if (i+1) % 1000 == 0:
            print('%d new images generated' % (i+1,))

    new_X = np.concatenate((orig_X, new_X))
    new_y = np.concatenate((orig_y, new_y))

    # Create dict of new data, and write it to disk via pickle file
    new_data = {'features': new_X, 'labels': new_y}
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
        images = X[start_idx: start_idx + batch_size]
        labels = y[start_idx: start_idx + batch_size]

        yield (np.array(images), np.array(labels))

        start_idx += batch_size


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
