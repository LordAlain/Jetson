from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
#import tensorflow
#from keras.layers.normalization import layer_normalization
import keras
print("Reg Train Keras version: ", keras.__version__)
import tensorflow as tf
print("Reg Train TF version: ", tf.__version__)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
# import cv2 as cv
import math
import os
import time
import pickle

import zipfile
from imageio import imread
# from skimage.io import imread
from skimage.transform import resize
import pprint


import matplotlib.pyplot as plt
#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

from utils import preprocess_data, next_batch, calculate_accuracy
from model import Model

print("Done.")
