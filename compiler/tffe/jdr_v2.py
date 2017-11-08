# Unit test for Tffe - single convolution


import tensorflow as tf
import numpy as np
import sys
import re

print("\nINFO: started as  ", " ".join(sys.argv))

dimStr = sys.argv[1]

# Sample dimStr : b1-h2-r2-s1-c4-m4-wmin-0.1-wmax0.1-imin1-imax5
dimStr = dimStr.upper() + "-"
if len(sys.argv) > 2:
  outPrefix = sys.argv[2]
else:
  outPrefix = "out_"
if len(sys.argv) > 3:
  netName = sys.argv[3]
else:
  netName = "jdr_v2"

dimList = re.split('([A-Z]+)(-?[\d\.]+)-', dimStr)
dimCmd = str(tuple(dimList[1::3])).replace("'", "") + " = " + str(tuple(map(float, dimList[2::3])))
dimCmd = dimCmd.replace(".0,", ",")
print(dimCmd)
assert(len(dimList[2::3]) == 10)
exec(dimCmd)

#(B, H, R, S, C, M) = (1,  4, 1, 1, 1, 1)
#(B, H, R, S, C, M) = (1,  8, 3, 1, 4, 4)
#(B, H, R, S, C, M) = (1, 12, 3, 1, 12, 12)
#(B, H, R, S, C, M) = (1, 16, 2, 1, 16, 16)
#(B, H, R, S, C, M) = (1, 16, 2, 1, 64, 64)
#(B, H, R, S, C, M) = (1, 16, 2, 1, 1, 1)
IF1 = np.zeros([B, H, H, C])
W1  = np.zeros([R, R, C, M])
#(WMIN, WMAX) = (-0.001,0.001)

strides = [1, S, S, 1]
padding = "SAME"


w1Values =  np.linspace(WMIN, WMAX, num=W1.size, dtype=np.float16).reshape(W1.shape)
print("w1\n", w1Values, "  ", w1Values.dtype)

w1 = tf.get_variable(name=netName+"/weight1",
                     initializer = w1Values, dtype=tf.float16)
i0 = tf.placeholder(tf.float16, shape=IF1.shape, name="input")

i1 = tf.nn.conv2d(i0, w1, strides, padding, name=netName + "/i1")
output = tf.identity(i1, name=netName+"/output")

i0val = np.linspace(IMIN, IMAX, num=IF1.size, dtype=np.float16).reshape(IF1.shape)
print("Inp=\n", i0val)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  res = sess.run(output, feed_dict={"input:0" : i0val})
  print("Res=\n", res)
  graph = tf.get_default_graph()
  tf.train.write_graph(graph, '.', outPrefix + 'graph.pb')
  saver = tf.train.Saver()
  prefixTFfix = ""
  if not outPrefix.startswith("/"):
    prefixTFfix = "./"
  saver.save(sess, prefixTFfix + outPrefix + "checkpoint.data")

