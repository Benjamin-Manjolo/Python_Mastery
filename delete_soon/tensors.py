#nd_array
#GPU support
#computational graph /Backpropagation
#immutable


import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"

#rank-1 tensor
rank_1 = tf.constant([1,1,3])
print(rank_1)

#rank-2 tensor
rank_2 = tf.constant([[1,1,3],[2,3,4]])
print(rank_2)

#ones filling
ones = tf.ones(3,3)
print(ones)

#random numbers filling
random = tf.random.normal((3,3),mean=0,stddev=1)
print(random)