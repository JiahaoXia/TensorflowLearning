import numpy as np
import tensorflow as tf

size = 10
pb_dir = 'pb_model/'
with tf.gfile.FastGFile(pb_dir + 'test-model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    X, pred = tf.import_graph_def(graph_def, return_elements=['input: 0', 'output: 0'])

z = tf.placeholder(name='new_label', shape=[None, 1], dtype=tf.float32)

new_beta = tf.get_variable(name='new_beta', shape=[1], initializer=tf.glorot_normal_initializer)
new_bias = tf.get_variable(name='new_bias', shape=[1], initializer=tf.glorot_uniform_initializer)

new_pred = tf.sigmoid(new_beta * pred + new_bias)

new_loss = tf.reduce_mean(tf.losses.log_loss(predictions=new_pred, labels=z))
train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9,
                                  beta2=0.999, epsilon=1e-8).minimize(new_loss)

batch_size = 8
feed_X = np.ones((batch_size, size)).astype(np.float32)
feed_z = np.array([[1],[1],[0],[0],[1],[1],[0],[0]]).astype(np.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(new_pred, feed_dict={X: feed_X}))
    sess.run(train_op, feed_dict={X: feed_X, z: feed_z})
    print(sess.run(new_pred, feed_dict={X: feed_X}))


