# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - convolution, (later) pool (CNN-like)


import tensorflow as tf
import numpy as np
import sys
import re

# To minimize likelihood of float16 overflow
# Example  0 1 2 3 4 5  =>  0 5 1 4 2 3
def permuteArr(arr):
  s = arr.size
  if s %2 == 0:
    a1 = arr.reshape(2, int(s/2))
    a1[1] = np.flip(a1[1], 0)
    a2 = a1.swapaxes(0, 1)
    a3 = a2.ravel()
  else:
    a3 = arr
  return(a3)

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
if len(sys.argv) > 4:
  dataType = sys.argv[4]
else:
  dataType = "float16"
# DataTypes
#   npDataType, tfDataType - for the data flow
#   fixedType - np.float16 - for generating inputs, weights
for t in ["np", "tf"]:
  exec("%sDataType = %s.%s" % (t, t, dataType))
fixedDataType = np.float16

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

wAllValues = permuteArr(np.linspace(WMIN, WMAX, num=(W1.size + W2.size), dtype=fixedDataType))
w1Values =  wAllValues[0:W1.size].reshape(W1.shape)
print("w1\n", w1Values, "  ", w1Values.dtype)
w2Values =  wAllValues[W1.size:W1.size+W2.size].reshape(W2.shape)
print("w2\n", w2Values, "  ", w2Values.dtype)

w1 = tf.get_variable(name=netName+"/weight1",
                     initializer = w1Values.astype(npDataType), dtype=tfDataType)
i0 = tf.placeholder(tfDataType, shape=IF1.shape, name="input")

i1 = tf.nn.conv2d(i0, w1, strides, padding, name=netName + "/i1")
#output = tf.nn.max_pool(i1, [1, P1, P1, 1], strides, padding, name=netName+"/output")
w2 = tf.get_variable(name=netName+"/weight2",
                     initializer = w2Values.astype(npDataType), dtype=tfDataType)
i2 = tf.nn.conv2d(i1, w2, strides, padding, name=netName + "/i2")
output = tf.identity(i2, name=netName+"/output")

i0val = permuteArr(np.linspace(IMIN, IMAX, num=IF1.size, dtype=fixedDataType)).reshape(IF1.shape)
np.save( outPrefix + 'ref_input.npy', i0val)
print("Inp=\n", i0val)

# Grow GPU memory as needed at the cost of fragmentation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  res = sess.run(output, feed_dict={"input:0" : i0val})
  print("Res=\n", res)
  print("INFO: the result contains %d infinite numbers" % (res.size - np.count_nonzero(np.isfinite(res))))
  graph = tf.get_default_graph()
  tf.train.write_graph(graph, '.', outPrefix + 'graph.pb')
  saver = tf.train.Saver()
  prefixTFfix = ""
  if not outPrefix.startswith("/"):
    prefixTFfix = "./"
  saver.save(sess, prefixTFfix + outPrefix + "checkpoint.data")

