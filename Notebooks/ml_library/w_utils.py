

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import copy


from ml_library.config import *
from ml_library.utils import *
from ml_library.model import *
# from ml_library.w_utils import *

def rain(images, eps):
    x = copy.copy(images)
    batch_size = images.shape[0]
    for i in range(batch_size):
        #         if np.mean(x[i]) > 100 and i < 1:
        #             x[i] = x[i] / (1+eps*0.15)
        for j in range(eps*40):
            np.random.seed(j)
            x_idx = np.random.randint(0, 32)
            y_idx = np.random.randint(0, 32)
            x[i, x_idx:x_idx+2, y_idx, :] = 0.8 * \
                x[i, x_idx:x_idx+2, y_idx, :] + 0.2*255
    x = np.clip(x, 0., 255.)
    return x


def fog(images, eps):
    x = copy.copy(images)
    batch_size = images.shape[0]
#     x = x / (eps*0.3+1) + eps * 10
    x = x * (1-eps*0.04) + eps * 7
#     for i in range(batch_size):
#         image = x[i].reshape(32, 32, 3)
#         temp = gaussian_filter(image, sigma=0.7)
#         x[i] = temp
    x = np.clip(x, 0., 255.)
    return x


def dark(images, eps):
    x = copy.copy(images)
    x = x - eps*10.
    x = np.clip(x, 0., 255.)
    return x


def light(images, eps):
    x = copy.copy(images)
    x = x + eps*10.
    x = np.clip(x, 0., 255.)
    return x


def blur(images, eps):
    batch_size = images.shape[0]
    x = np.zeros(images.shape)
    for i in range(batch_size):
        image = images[i].reshape(32, 32, 3)
        temp = gaussian_filter(image, sigma=0.5+eps*0.1)
        x[i] = temp
    return x


def ocul(images, eps):
    batch_size = images.shape[0]
    x = copy.copy(images)
    for i in range(batch_size):
        image = images[i].reshape(32, 32, 3)
        idx = np.random.randint(0, 32-eps)
        x[i, idx:idx+eps, idx:idx+eps, :] = 0.
#         x[i] = image
    return x
