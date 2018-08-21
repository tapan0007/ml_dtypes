# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - one convolution followed by one pool


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
# h4-MaxPool-k1  returns "MaxPull","h4-k1"
def getPoolPadType(confStr):
  conf = {}
  padType = None
  poolType = None
  usePerm = False
  for s in ["MaxPool", "AvgPool"]:
    s1 = "-" + s
    if s1 in confStr:
      confStr = confStr.replace(s1, "")
      poolType = s
      break
  for s in ["SAME", "VALID"]:
    s1 = "-" + s
    if s1 in confStr:
      confStr = confStr.replace(s1, "")
      padType = s
      break
  for s in ["PERM"]:
    s1 = "-" + s
    if s1 in confStr:
      confStr = confStr.replace(s1, "")
      usePerm = True
      break
  return poolType,padType,usePerm,confStr

print("\nINFO: started as  ", " ".join(sys.argv))

dimStr = sys.argv[1]

# Sample dimStr : tfloat16-b1-h4-r1-s1-c1-m1-SAME-MaxPool-k2-d1-wmin2-wmax2-imin1-imax16
#  k pool kernel, d pool stride
poolType, padType, usePerm, dimStr = getPoolPadType(dimStr)
dimStr = dimStr.upper() + "-"
if len(sys.argv) > 2:
  outPrefix = sys.argv[2]
else:
  outPrefix = "out_"
if len(sys.argv) > 3:
  netName = sys.argv[3]
else:
  netName = "jdr_v4"
if len(sys.argv) > 4:
  dataType = sys.argv[4]
else:
  dataType = "float16"
# DataTypes
#   npDataType, tfDataType - for the data flow
#   fixedType - np.float16 - for generating inputs, weights
for t in ["np", "tf"]:
  exec("%sDataType = %s.%s" % (t, t, dataType))

dimList = re.split('([A-Z]+)(-?[\d\.]+)-', dimStr)
dimCmd = str(tuple(dimList[1::3])).replace("'", "") + " = " + str(tuple(map(float, dimList[2::3])))
dimCmd = dimCmd.replace(".0,", ",")
print(dimCmd)
assert(len(dimList[2::3]) == 13)
exec(dimCmd)
if (IMIN >= IMAX):
    print("Warning: Imin >= Imax:", IMIN< IMAX)
if (WMIN >= WMAX):
    print("Warning: Wmin >= Wmax:", WMIN< WMAX)

# Conv
IF1shapeNHWC = [B, H, H, C]
IF1 = np.zeros(IF1shapeNHWC)
convKernelNHWC = [1, R, R, 1]
convStrides = [1, S, S, 1]
i0val = np.linspace(IMIN, IMAX, num=IF1.size, dtype=npDataType)
if usePerm:
  i0val = permuteArr(i0val)
i0val = i0val.reshape(IF1shapeNHWC)

# Weight
W1shapeRSCM = [R, R, C, M]
W1 = np.zeros(W1shapeRSCM)
W1val = np.linspace(WMIN, WMAX, num=W1.size, dtype=npDataType)
if usePerm:
  W1val = permuteArr(W1val)
W1val = W1val.reshape(W1shapeRSCM)

W3shapeRSCM = [R, R, M, M]
W3 = np.zeros(W3shapeRSCM)
W3val = np.linspace(WMIN, WMAX, num=W3.size, dtype=npDataType)
if usePerm:
  W3val = permuteArr(W3val)
W3val = W3val.reshape(W3shapeRSCM)

# Pool
poolKernelNHWC = [1, K, K, 1]
poolStrides = [1, D, D, 1]

i0 = tf.placeholder(tfDataType, shape=IF1.shape, name="input")

w1 = tf.get_variable(name=netName+"/weight1",
                             initializer = W1val.astype(npDataType), dtype=tfDataType)
i1 = tf.nn.conv2d(i0, w1, convStrides, 'SAME', name=netName + "/i1")

i1a = tf.nn.relu(i0, name=netName + "/i1a_relu")

if poolType == "AvgPool":
  i2 = tf.nn.avg_pool(i1a, poolKernelNHWC, poolStrides, padType, name=netName + "/i2")
else:
  i2 = tf.nn.max_pool(i1a, poolKernelNHWC, poolStrides, padType, name=netName + "/i2")

w3 = tf.get_variable(name=netName+"/weight3",
                             initializer = W3val.astype(npDataType), dtype=tfDataType)
i3 = tf.nn.conv2d(i2, w3, [1, 1, 1, 1], 'SAME', name=netName + "/i3")

output = tf.identity(i3, name=netName+"/output")

i0val = np.linspace(IMIN, IMAX, num=IF1.size, dtype=npDataType).reshape(IF1.shape)
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


