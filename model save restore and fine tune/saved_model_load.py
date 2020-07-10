import tensorflow as tf
import numpy as np

size = 10

batch_size = 8
feed_X = np.ones((batch_size, size)).astype(np.float32)
feed_y = np.ones((8, 1)).astype(np.float32)

# 1. know tensor name
saved_model_dir = 'saved_model/1/'
# with tf.Session() as sess:
#     meta_graph_def = tf.saved_model.loader.load(sess, tags=['serve'], export_dir=saved_model_dir)
#     graph = tf.get_default_graph()
#     X = graph.get_tensor_by_name('input: 0')
#     pred = graph.get_tensor_by_name('output: 0')
#     print(sess.run(pred, feed_dict={X: feed_X}))

# 2. do not know tensor name
with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, tags=['serve'], export_dir=saved_model_dir)
    signature = meta_graph_def.signature_def
    print(signature)
    X = signature['serving_default'].inputs['input'].name
    pred = signature['serving_default'].outputs['output'].name
    print(sess.run(pred, feed_dict={X: feed_X}))