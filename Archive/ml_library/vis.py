# Imports
import matplotlib.pyplot as plt
import numpy as np
from Notebooks.ml_library.utils import map_labels, split

from config import *
from utils import *
from model import *

# Load data
training_file = "./Datasets/GTSRB_Final_Training_Images.zip"
testing_file = "./Datasets/GTSRB_Final_Test_Images.zip"

train = generateTensor(training_file)
test = generateTensor(testing_file)

x_train, y_train = split(train)
x_test, y_test = split(test)

label_map = map_labels('signnames.csv')


# To start off let's do a basic data summary.

# Number of training examples
n_train = x_train.shape[0]

# Number of testing examples
n_test = x_test.shape[0]

# What's the shape of an image?
image_shape = x_train.shape[0:]

# How many classes are in the dataset
n_classes = y_train.shape

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Randomly choose indices to represent which datapoints we choose from the training set
num_images = 3
indices = np.random.choice(
    list(range(n_train)), size=num_images, replace=False)

# Obtain the images and labels
images = x_train[indices]
labels = y_train[indices]

# Display the images
plt.rcParams["figure.figsize"] = [15, 5]

for i, image in enumerate(images):
    plt.subplot(1, num_images, i+1)
    plt.imshow(image)
    plt.title(label_map[labels[i]])

plt.tight_layout()
plt.show()


'''

# Count frequency of Training
labels, counts = np.unique(y_train, return_counts=True)

# Plot the histogram
plt.rcParams["figure.figsize"] = [15, 5]
axes = plt.gca()
axes.set_xlim([-1,43])

plt.bar(labels, counts, tick_label=labels, width=0.8, align='center')
plt.title('Class Distribution across Training Data')
plt.show()



# Count frequency of Testing
labels, counts = np.unique(y_test, return_counts=True)

# Plot the histogram
plt.rcParams["figure.figsize"] = [15, 5]
axes = plt.gca()
axes.set_xlim([-1,43])

plt.bar(labels, counts, tick_label=labels, width=0.8, align='center')
plt.title('Class Distribution across Test Data')
plt.show()

'''

# # Load augmented training dataset
# with open('train_aug.p', mode='rb') as f:
#     train = pickle.load(f)


train_aug = data_aug(train)
# test_aug = data_aug(test)


x_train_aug, y_train_aug = train['features'], train['labels']

# Count frequency of each label
labels, counts = np.unique(y_train_aug, return_counts=True)

# Plot the histogram
plt.rcParams["figure.figsize"] = [15, 5]
axes = plt.gca()
axes.set_xlim([-1, 43])

plt.bar(labels, counts, tick_label=labels, width=0.8, align='center')
plt.title('Class Distribution across Augmented Training Data')
plt.show()


# Choose random training image and visually inspect transformed images
random_idx = np.random.randint(0, n_train)
image = x_train_aug[random_idx]

for i in range(9):
    rand_idx = np.random.randint(x_train_aug.shape[0])
    image = x_train_aug[rand_idx]
    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    plt.title('Image at Idx %d' % (rand_idx,))

plt.tight_layout()
plt.show()
