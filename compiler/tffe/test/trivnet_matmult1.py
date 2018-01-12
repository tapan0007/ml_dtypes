# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - matrix multiply (LSTM)


import tensorflow as tf
import numpy as np

strides = [1, 1, 1, 1]
padding = "SAME"

k = 32
netName = "jdr_v1"

w1Values =  np.arange(2*k, dtype=np.float16).reshape(1, k, 2)
print("w1\n", w1Values, "  ", w1Values.dtype)

w1 = tf.get_variable(name=netName+"/weight1",
                     initializer = w1Values, dtype=tf.float16)
i0 = tf.placeholder(tf.float16, shape=(1,2,k), name="input")

output = tf.matmul(i0, w1, name=netName + "/output")
#i1a = tf.reshape(i1, (1,2,2,1))
#i2 = tf.nn.max_pool(i1a, [0, 4, 4, 0], strides, padding, name=netName+"/i2")


i0val = np.arange(2*k, dtype=np.float16).reshape(1, 2, k)
print("Inp=\n", i0val)

# Grow GPU memory as needed at the cost of fragmentation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  res = sess.run(output, feed_dict={"input:0" : i0val})
  print("Res=\n", res)
  graph = tf.get_default_graph()
  tf.train.write_graph(graph, '.', 'out_jdr_v1.pb')
  saver = tf.train.Saver()
  saver.save(sess,"./out_jdr_v1.data")

