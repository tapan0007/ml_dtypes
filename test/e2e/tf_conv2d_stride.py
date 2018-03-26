# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Test to debug cpu vs gpu difference in conv2d with large stride

# To run:
#  gpu -> this looks correct
#    python3 tf_conv2d_stride.py
#      2   18
#    258  274
#  cpu
#    ( setenv CUDA_VISIBLE_DEVICES ; python3 tf_conv2d_stride.py)
#     104  120
#     360  376

import tensorflow as tf
import numpy as np
import sys
import re

np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180

tfDataType = np.float16
(B, H, C, R, M, S) = ( 1, 16, 1, 1, 1, 8)
W1  = np.zeros([R, R, C, M])
IF1 = np.zeros([B, H, H, C])
strides = [1, S, S, 1]
padding = "SAME"

w1val = np.linspace(2, R*R+1, num=W1.size,  dtype=tfDataType).reshape(W1.shape)
i0val = np.linspace(1, H*H,   num=IF1.size, dtype=tfDataType).reshape(IF1.shape)

w1 = tf.get_variable(name="weight1",
                     initializer = w1val, dtype=tfDataType)
i0 = tf.placeholder(tfDataType, shape=IF1.shape, name="input")

i1 = tf.nn.conv2d(i0, w1, strides, padding, name="i1")
output = tf.identity(i1, name="output")

# Grow GPU memory as needed at the cost of fragmentation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  res = sess.run(output, feed_dict={"input:0" : i0val})
  print("Input=\n", i0val.reshape(H, H),
        "\n\nWeight=\n", w1val.reshape(R, R),
        "\n\nRes=\n", res.reshape(H // S, H // S))
  graph = tf.get_default_graph()
