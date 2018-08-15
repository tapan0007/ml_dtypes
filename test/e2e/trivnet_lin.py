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

# Extract text-only config options from the config string
# l10-relu-tanh-b1 => {"relu" : 1, "tanh" : 1}, return letover "l10-b1"
def getConfOpts(confStr):
  conf = {}
  for s in ["RELU", "TANH", "SOFTPLUS", "SIGMOID"]:
    s1 = "-" + s
    if s1 in confStr:
      confStr = confStr.replace(s1, "")
      conf[s] = True
  return(conf, confStr)

print("\nINFO: started as  ", " ".join(sys.argv))

confStr = sys.argv[1]

# Sample confStr : tfloat16-l2-b1-h2-r2-s1-c4-m4-wmin-0.1-wmax0.1-imin1-imax5
confStr = confStr.upper() + "-"
if len(sys.argv) > 2:
  outPrefix = sys.argv[2]
else:
  outPrefix = "out_"
if len(sys.argv) > 3:
  netName = sys.argv[3]
else:
  netName = "jdr_v2"
(conf, dimStr) = getConfOpts(confStr)
print("INFO: options config ", conf)

dimList = re.split('([A-Z]+)(-?[\d\.]+)-', dimStr)
dimCmd = str(tuple(dimList[1::3])).replace("'", "") + " = " + str(tuple(map(float, dimList[2::3])))
dimCmd = dimCmd.replace(".0,", ",")

print(dimCmd)
assert(len(dimList[2::3]) == 12)
exec(dimCmd)
if (IMIN >= IMAX):
    print("Warning: Imin >= Imax:", IMIN< IMAX)
if (WMIN >= WMAX):
    print("Warning: Wmin >= Wmax:", WMIN< WMAX)
assert(C == M)  # Two back to back convolutions must have same number of channels (till pooling is added)
assert(L > 0)

# TO_DO - add support for int types
dataType = TFLOAT
# DataTypes - likely legacy, we are heading toward clean 32b or clean 16b or quantized 32/8
#   npDataType, tfDataType - for the data flow
#   fixedDataType - np.float16 - for generating inputs, weights
for t in ["np", "tf"]:
  exec("%sDataType = %s.float%s" % (t, t, dataType))
fixedDataType = npDataType


IF1 = np.zeros([B, H, H, C])
W1  = np.zeros([R, R, C, M])

strides = [1, S, S, 1]
padding = "SAME"

numConvLayers = L
wAllValues = permuteArr(np.linspace(WMIN, WMAX, num=(W1.size * numConvLayers), dtype=fixedDataType))

layers = []
weights = []
layers.append(tf.placeholder(tfDataType, shape=IF1.shape, name="input"))
for layerId in range(1, numConvLayers):
  wValues =  wAllValues[(layerId-1)*W1.size:layerId * W1.size].reshape(W1.shape)
  w = tf.get_variable(name=netName+"/weight" + str(layerId),
                             initializer = wValues.astype(npDataType), dtype=tfDataType)
  weights.append(w)
  op = tf.nn.conv2d(layers[-1], weights[layerId - 1], strides, padding, name=netName + "/conv" + str(layerId))
  layers.append(op)
  # Relu
  if (conf.get("RELU")):
    op = tf.nn.relu(layers[-1], name=netName + "/relu" + str(layerId))
    layers.append(op)
  if (conf.get("TANH")):
    op = tf.nn.tanh(layers[-1], name=netName + "/tanh" + str(layerId))
    layers.append(op)
  if (conf.get("SOFTPLUS")):
    op = tf.nn.softplus(layers[-1], name=netName + "/softplus" + str(layerId))
    layers.append(op)
  if (conf.get("SIGMOID")):
    op = tf.nn.sigmoid(layers[-1], name=netName + "/sigmoid" + str(layerId))
    layers.append(op)
output = tf.identity(layers[-1], name=netName+"/output")

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

