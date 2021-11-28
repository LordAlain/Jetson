'''
Augment the data
'''
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
# import cv2 as cv
import pickle

from config import *
from utils import *

# Image data augmentation parameters
ANGLE = 15
TRANSLATION = 0.2
WARP = 0.0  # 0.05
#NUM_NEW_IMAGES = 100000
NUM_NEW_IMAGES = 1000

########################################################
# Helper functions
########################################################


########################################################
# Main function
########################################################
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


if __name__ == '__main__':
    # This part is for visualization and/or debug
    # with open('train.p', mode='rb') as f:
    #	orig_data = pickle.load(f)
    # display_random_images(orig_data['features'])

    # This actually creates the augmented dataset
    data_aug('train.p', 'train_aug.p', NUM_NEW_IMAGES)

    # For debug, display random images from augmented dataset
    # display_random_aug('train_aug.p')
