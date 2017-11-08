# Unit test for Tffe - convolution, (later) pool (CNN-like)


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
assert(C == M)  # Two back to back convolutions must have same number of channels (till pooling is added)

IF1 = np.zeros([B, H, H, C])
W1  = np.zeros([R, R, C, M])
W2  = np.zeros([R, R, C, M])

strides = [1, S, S, 1]
padding = "SAME"

wAllValues = np.linspace(WMIN, WMAX, num=(W1.size + W2.size), dtype=np.float16)
w1Values =  wAllValues[0:W1.size].reshape(W1.shape)
print("w1\n", w1Values, "  ", w1Values.dtype)
w2Values =  wAllValues[W1.size:W1.size+W2.size].reshape(W2.shape)
print("w2\n", w2Values, "  ", w2Values.dtype)

w1 = tf.get_variable(name=netName+"/weight1",
                     initializer = w1Values, dtype=tf.float16)
i0 = tf.placeholder(tf.float16, shape=IF1.shape, name="input")

i1 = tf.nn.conv2d(i0, w1, strides, padding, name=netName + "/i1")
#output = tf.nn.max_pool(i1, [1, P1, P1, 1], strides, padding, name=netName+"/output")
w2 = tf.get_variable(name=netName+"/weight2",
                     initializer = w2Values, dtype=tf.float16)
i2 = tf.nn.conv2d(i1, w2, strides, padding, name=netName + "/i2")
output = tf.identity(i2, name=netName+"/output")

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

