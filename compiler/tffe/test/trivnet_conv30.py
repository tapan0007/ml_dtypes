# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - convolution, (later) pool (CNN-like)

# Samples
# Depth 30
# make -f $KAENA_PATH/compiler/tffe/Makefile jdr_v4 NN_CONFIG=b1-h4-r3-s1-c1-m1-wmin-0.2-wmax0.2-imin-10000-imax10000 OUT_PREFIX=trivnet_ NN_NAME=150conv
#

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

strides = [1, S, S, 1]
padding = "SAME"

numLayers = 6
wAllValues = permuteArr(np.linspace(WMIN, WMAX, num=(W1.size * numLayers), dtype=fixedDataType))

layers = []
weights = []
layers.append(tf.placeholder(tfDataType, shape=IF1.shape, name="input"))
for layerId in range(1, numLayers):
  wValues =  wAllValues[(layerId-1)*W1.size:layerId * W1.size].reshape(W1.shape)
  w = tf.get_variable(name=netName+"/weight" + str(layerId),
                             initializer = wValues.astype(npDataType), dtype=tfDataType)
  weights.append(w)
  op = tf.nn.conv2d(layers[layerId - 1], weights[layerId - 1], strides, padding, name=netName + "/conv" + str(layerId))
  layers.append(op)
output = tf.identity(layers[numLayers - 1], name=netName+"/output")

i0val = permuteArr(np.linspace(IMIN, IMAX, num=IF1.size, dtype=fixedDataType)).reshape(IF1.shape)
np.save( outPrefix + 'ref_input.npy', i0val)
print("Inp=\n", i0val)
with tf.Session() as sess:
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

