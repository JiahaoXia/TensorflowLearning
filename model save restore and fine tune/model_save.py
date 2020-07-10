# save trained TensorFlow model
# Reference: https://zhuanlan.zhihu.com/p/128546377
# Date: 2020-07-10

import tensorflow as tf
import numpy as np

size = 10

# create input
X = tf.placeholder(name="input", shape=[None, size], dtype=tf.float32)
y = tf.placeholder(name="label", shape=[None, 1], dtype=tf.float32)

# graph
beta = tf.get_variable(name="beta", shape=[size, 1],
                       initializer=tf.glorot_normal_initializer)
bias = tf.get_variable(name="bias", shape=[1],
                       initializer=tf.glorot_normal_initializer)
pred = tf.add(tf.matmul(X, beta), bias, name="output")

# loss function
loss = tf.losses.mean_squared_error(y, pred)

# train operation
train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9,
                                  beta2=0.999, epsilon=1e-8).minimize(loss)

# create training data
batch_size = 8
feed_X = np.ones((batch_size, size)).astype(np.float32)
feed_y = np.ones((8, 1)).astype(np.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(pred, feed_dict={X: feed_X}))
    sess.run(train_op, feed_dict={X: feed_X, y: feed_y})
    print(sess.run(pred, feed_dict={X: feed_X}))












