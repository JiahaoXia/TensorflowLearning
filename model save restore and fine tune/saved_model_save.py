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

version = '1/'
saved_model_dir = 'saved_model/'
builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir + version)

signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'input': tf.saved_model.utils.build_tensor_info(X)},
    outputs={'output': tf.saved_model.utils.build_tensor_info(pred)},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    builder.add_meta_graph_and_variables(sess,
                                         tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={'serving_default': signature})
    builder.save()