# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Unit test for Tffe - biasadd alone
# The weight ranges are used for the bias add


import tensorflow as tf
import numpy as np
import sys
import re

print("\nINFO: started as  ", " ".join(sys.argv))

dimStr = sys.argv[1]

# Sample dimStr : tfloat16-b1-h2-c4-wmin-0.1-wmax0.1-imin1-imax5
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
assert(len(dimList[2::3]) == 8)
exec(dimCmd)

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


IF1 = np.zeros([B, H, H, C])
BIAS1 = np.zeros([C])
w = tf.get_variable(name=netName+"/bias1",
                    initializer = np.linspace(WMIN, WMAX, num=BIAS1.size, dtype=npDataType).reshape(BIAS1.shape),
                    dtype=tfDataType)
i0 = tf.placeholder(tfDataType, shape=IF1.shape, name="input")
i1 = tf.nn.bias_add(i0, w, name=netName + "/i1")
output = tf.identity(i1, name=netName+"/output")

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
