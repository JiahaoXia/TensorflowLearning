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

# save
pb_dir = 'pb_model/'
saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    graph_def = tf.get_default_graph().as_graph_def()
    var_list = ['input', 'label', 'beta', 'bias', 'output']
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, var_list)
    with tf.gfile.FastGFile(pb_dir + 'test-model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())