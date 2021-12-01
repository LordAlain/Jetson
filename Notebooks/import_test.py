import warnings

# Imports
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.python.ops.gen_math_ops import AccumulateNV2
# import cv2 as cv
# from skimage import io, transform

# import PIL
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# Imports
# import tensorflow as tf
# import cupy as cp
# import numpy as np
# from tensorflow import keras
# import PIL
# from pillow import Image
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# %matplotlib inline
# from tensorflow.keras import layers
# from tensorflow.keras import Input
# from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import *
# import tempfile

# from ml_library.config import *
from ml_library.utils import *
# from ml_library.model import *
# from ml_library.w_utils import *
import ssl

# print("Reg Train Keras version: ", keras.__version__)
# print("Reg Train TF version: ", tf.__version__)


# warnings.filterwarnings("ignore")

# model_dir = '/models/'
# tmpdir = tempfile.mkdtemp()
# print("Tmpdir = ", tmpdir)
# print("Training file: ", os.path.exists(
#     "./Datasets/GTSRB_Final_Training_Images.zip"))
# print("Testing file: ", os.path.exists(
#     "./Datasets/GTSRB_Final_Test_Images.zip"))
#url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
#url = "http://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html"
# filename = "./Datasets/_GTSRB_Final_Test_GT.zip"
# ssl._create_default_https_context = ssl._create_unverified_context
# # urllib.request.urlretrieve(url)
# urllib.request.urlretrieve(url, filename)
# # "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

importDatasets()

print("Done.")
