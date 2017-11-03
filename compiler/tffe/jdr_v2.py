# Unit test for Tffe - single convolution


import tensorflow as tf
import numpy as np

netName = "jdr_v2"
#(B1, H1, R1, C1, M1) = (1, 8, 3, 4, 4)
(B1, H1, R1, C1, M1) = (1, 4, 1, 1, 1)
IF1 = np.zeros([B1, H1, H1, C1])
W1  = np.zeros([R1, R1, C1, M1])
(Wmin, Wmax) = (1,1)

strides = [1, 1, 1, 1]
padding = "SAME"


w1Values =  np.linspace(Wmin, Wmax, num=W1.size, dtype=np.float16).reshape(W1.shape)
print("w1\n", w1Values, "  ", w1Values.dtype)

w1 = tf.get_variable(name=netName+"/weight1",
                     initializer = w1Values, dtype=tf.float16)
i0 = tf.placeholder(tf.float16, shape=IF1.shape, name="input")

i1 = tf.nn.conv2d(i0, w1, strides, padding, name=netName + "/i1")
output = tf.identity(i1, name=netName+"/output")

i0val = np.linspace(0, 1, num=IF1.size, dtype=np.float16).reshape(IF1.shape)
print("Inp=\n", i0val)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  res = sess.run(output, feed_dict={"input:0" : i0val})
  print("Res=\n", res)
  graph = tf.get_default_graph()
  tf.train.write_graph(graph, '.', 'out_jdr_v2.pb')
  saver = tf.train.Saver()
  saver.save(sess,"./out_jdr_v2.data")

