# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - single convolution


import tensorflow as tf
import numpy as np
import sys
import re
from tensorflow.python.ops import array_ops

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

# Extract custom config
# Example for filling specific values for FMAP F(31,31)=3
def getConfFmap(confStr):
  conf = []
  for regexp in [r'-F_(\d+)_(\d+)=(\d+)']:
    match = re.search(regexp, confStr)
    if match:
      conf.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
      confStr = confStr.replace(match.group(0), "")
  return(conf, confStr)

print("\nINFO: started as  ", " ".join(sys.argv))

dimStr = sys.argv[1]
fmapValList,dimStr = getConfFmap(dimStr)

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
print ("dimStr = ",dimStr)
print ("dimList = ",dimList)
dimCmd = str(tuple(dimList[1::3])).replace("'", "") + " = " + str(tuple(map(float, dimList[2::3])))
dimCmd = dimCmd.replace(".0,", ",")
print(dimCmd)
#assert(len(dimList[2::3]) == 1)
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

#mm = 1
#print ("mm = %d"%mm)
IF1 = np.zeros([B, H, H, C])
IF2 = np.zeros([B, H, H, C])
W = []
w_size = 0
for i in range(int(NI)):
    mm = np.random.randint(low = (-M + 1), high = (2*M))
    W.append(np.zeros([R, R, C, M + mm]))
    w_size += W[-1].size
#    print ("w_size = %d"%w_size)
#W2  = np.zeros([R, R, C, M + mm])
#W3  = np.zeros([R, R, 2*M + cc, M])
#Wmove = np.zeros([2*C, R, R, M])

strides = [1, S, S, 1]
padding = "SAME"


wAllValues = permuteArr(np.linspace(WMIN, WMAX, num=w_size, dtype=npDataType))
wValues = []
loc = 0
w = []
for i in range(int(NI)):
#    print ("loc = %d W[%d].size = %d W[%d].shape = "\
#           %(loc, i, W[i].size, i), W[i].shape)
#    print ("wAllValues[%d:%d] = "\
#           %(loc, W[i].size), wAllValues[loc:loc+W[i].size])
    wValues.append(wAllValues[loc:loc+W[i].size].reshape(W[i].shape))
    w.append(tf.get_variable(name=netName+"/weight"+str(i),
                             initializer = wValues[i].astype(npDataType),\
                             dtype=tfDataType))
#    print("w%d\n"%i, wValues[i], "  ", wValues[i].dtype)
    loc += W[i].size

#w1 = tf.get_variable(name=netName+"/weight1",
#                     initializer = w1Values.astype(npDataType),\
#                     dtype=tfDataType)
#w2 = tf.get_variable(name=netName+"/weight2",
#                     initializer = w2Values.astype(npDataType),\
#                     dtype=tfDataType)
#w3 = tf.get_variable(name=netName+"/weight3",
#                     initializer = w3Values.astype(npDataType),\
#                     dtype=tfDataType)
#wmove = tf.get_variable(name=netName+"/weight_move",
#                     initializer = wmoveValues.astype(npDataType),\
#                     dtype=tfDataType)
#i0 = tf.placeholder(tfDataType, shape=IF1.shape, name="input0")
#i1 = tf.placeholder(tfDataType, shape=IF2.shape, name="input1")
i = tf.placeholder(tfDataType, shape=IF1.shape, name="input")
ii = []
for conv_cnt in range(int(NI)):
    ii.append(tf.nn.conv2d(i\
                           , w[conv_cnt]\
                           , strides\
                           , padding\
                           , name=netName + "/i"+str(conv_cnt)\
                          )\
             )
#i1 = tf.nn.conv2d(i, w1, strides, padding, name=netName + "/i1")
#i2 = tf.nn.conv2d(i, w2, strides, padding, name=netName + "/i2")

#i3 = array_ops.concat([i1, i2], -1, name=netName+"/i3")
i3 = array_ops.concat(ii, -1, name=netName+"/i3")
#i4 = tf.nn.conv2d(i3, w3, strides, padding, name=netName + "/i4")
output = tf.identity(i3, name=netName+"/output")

#i0val = permuteArr(np.linspace(IMIN, IMAX, num=IF1.size, dtype=npDataType)).reshape(IF1.shape)
np.random.seed(17)
i0val = np.random.random(IF1.shape)
#i1val = np.random.random(IF2.shape)

# Overide the values:
for row,col,val in fmapValList:
  i0val[0, row, col, 0] = val

np.save( outPrefix + 'ref_input.npy', i0val)
#np.save( outPrefix + 'ref_weight_move.npy', wmoveValues)
#np.save( outPrefix + 'ref_input1.npy', i1val)
print("Inp0=\n", i0val)
#print("wmoveValue=\n", wmoveValues)
#print("Inp1=\n", i1val)

# Grow GPU memory as needed at the cost of fragmentation.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
#res = sess.run(output, feed_dict={"input0:0" : i0val, "input1:0" : i1val})
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
