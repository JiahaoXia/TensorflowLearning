from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

print("Setting Eager mode...")
tf.enable_eager_execution()
tfe = tf.contrib.eager

print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)

print("Running operations, without tf.Session")
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)

print("Mixing operations with Tensors and Numpy Arrays")
a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
b = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)