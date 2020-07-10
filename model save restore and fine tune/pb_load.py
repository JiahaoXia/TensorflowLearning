import tensorflow as tf
import numpy as np

size = 10
pb_dir = 'pb_model/'
with tf.gfile.FastGFile(pb_dir + 'test-model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    X, pred = tf.import_graph_def(graph_def, return_elements=['input: 0', 'output: 0'])

# create training data
batch_size = 8
feed_X = np.ones((batch_size, size)).astype(np.float32)
feed_y = np.ones((8, 1)).astype(np.float32)

with tf.Session() as sess:
    print(sess.run(pred, feed_dict={X: feed_X}))
