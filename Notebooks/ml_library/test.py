# Imports
import tensorflow as tf
import math

from config import *
from utils import *
from model import *

model = Model()

x, y, keep_prob, logits, optimizer, predictions, accuracy = model.x, model.y, model.keep_prob, model.logits, model.optimizer, model.predictions, model.accuracy

# with open('accuracy_history.p', mode='rb') as f:
# 	accuracy_history = pickle.load(f)

# hist = np.transpose(np.array(accuracy_history))
# plt.plot(hist[0], 'b')  # training accuracy
# plt.plot(hist[1], 'r')  # validation accuracy
# plt.title('Train (blue) and Validation (red) Accuracies over Epochs')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.show()

# Load Data
label_map = map_labels('signnames.csv')
testing_file = './Datasets/GTSRB_Final_Test_Images.zip'
test = generateTensor(testing_file)
X_test, y_test = preprocess_data(test)

with tf.Session() as sess:

    saver = tf.train.Saver()
    filename = tf.train.latest_checkpoint("./model/")
    print("Latest training checkpoint is ", filename)
    if filename != None:
        saver.restore(sess, filename)
    else:
        print("No checkpoint found, exit.")
        exit()

    Accuracy = 0
    num = 10
    batch_size = math.ceil(X_test.shape[0] / num)
    test_gen = next_batch(X_test, y_test, batch_size, True)

    for i in range(num):

        # Run testing on each batch
        images, labels = next(test_gen)

        # Perform gradient update (i.e. training step) on current batch
        acc = sess.run(model.accuracy, feed_dict={
                       x: images, y: labels, keep_prob: 1})

        print("Test iter", i, "acc :", acc)
        Accuracy += acc
        # idx = sess.run(model.predictions, feed_dict={x: images, y: labels, keep_prob: 1})
        # print(label_map[idx[0]])

    Accuracy /= num
    print("Overall acc :", acc)
