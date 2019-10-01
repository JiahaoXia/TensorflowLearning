import tensorflow as tf
hello = tf.constant('hello, tensorflow!')

# start tf session
sess = tf.Session()

# run graph
print(sess.run(hello))