# Import MNIST
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Load data
X_train = mnist.train.images
Y_train = mnist.train.labels

X_test = mnist.test.images
Y_test = mnist.test.labels

a = np.array([1, 2, 3, 4, 5])
print(a)
b = np.argmax(a)
print(b)

print("***done***")