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
# h4-PERM-k1  returns "PERM","h4-k1"
def getPadType(confStr):
  conf = {}
  padType = None
  poolType = None
  usePerm = False
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
poolType, padType, usePerm, dimStr = getPadType(dimStr)
dimStr = dimStr.upper() + "-"
if len(sys.argv) > 2:
  outPrefix = sys.argv[2]
else:
  outPrefix = "out_"
if len(sys.argv) > 3:
  netName = sys.argv[3]
else:
  netName = "jdr_v5"

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

# DataTypes
#   npDataType, tfDataType - for the data flow
dataType = "float16"
try:
  dataType = "float%d" % TFLOAT
except:
  try:
    dataType = "int%d" % TINT
  except:
    try:
      dataType = "uint%d" % TUINT
    except:
      print("ERROR: no known type, check your t... section of the config string")
      exit(1)
for t in ["np", "tf"]:
  exec("%sDataType = %s.%s" % (t, t, dataType))



# Conv
IF1shapeNHWC = [B, C]
IF1 = np.zeros(IF1shapeNHWC)
i0val = np.linspace(IMIN, IMAX, num=IF1.size, dtype=npDataType)
if usePerm:
  i0val = permuteArr(i0val)
i0val = i0val.reshape(IF1shapeNHWC)

# Weight
W1shapeRSCM = [C, M]
W1 = np.zeros(W1shapeRSCM)
W1val = np.linspace(WMIN, WMAX, num=W1.size, dtype=npDataType)
if usePerm:
  W1val = permuteArr(W1val)
W1val = W1val.reshape(W1shapeRSCM)

i0 = tf.placeholder(tfDataType, shape=IF1.shape, name="input")

# branch 0
w1 = tf.get_variable(name=netName+"/weight1",
                             initializer = W1val.astype(npDataType), dtype=tfDataType)
i1 = tf.matmul(i0, w1, name=netName + "/i1")

# branch 1
w1b = tf.get_variable(name=netName+"/weight1b",
                             initializer = W1val.astype(npDataType), dtype=tfDataType)
i1b = tf.matmul(i0, w1b, name=netName + "/i1b")

# branch 1
w1c = tf.get_variable(name=netName+"/weight1c",
                             initializer = W1val.astype(npDataType), dtype=tfDataType)
i1c = tf.matmul(i0, w1c, name=netName + "/i1c")

# branch 1
w1d = tf.get_variable(name=netName+"/weight1d",
                             initializer = W1val.astype(npDataType), dtype=tfDataType)
i1d = tf.matmul(i0, w1d, name=netName + "/i1d")

#w1c = tf.get_variable(name=netName+"/weight1c",
#                             initializer = W1val.astype(npDataType), dtype=tfDataType)
#i1c = tf.nn.conv2d(i2b, w1c, convStrides, padType, name=netName + "/i1c")

#ba1c = tf.get_variable(name=netName+"/bias1c",
#                      initializer = BAval.astype(npDataType), dtype=tfDataType)
#i2c = tf.nn.bias_add(i1c, ba1c, name=netName + "/i2c")

# add
i3 = tf.add(i1, i1b, name=netName + "/i3")
i3b = tf.add(i3, i1c, name=netName + "/i3b")
i3c = tf.add(i3b, i1d, name=netName + "/i3c")

output = tf.identity(i3c, name=netName+"/output")


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

