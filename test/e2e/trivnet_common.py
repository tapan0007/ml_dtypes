"""
Copyright (C) 2017, Amazon.com. All Rights Reserved
"""
"""Common file for Trinet tests

To use, import everything in this file
import trivnet_common *

argv[1]:

    Configuration string is a string of parameter/value pairs separated by dash.
    Additionally, the data type is specified as t<type>.

    It is passed as the first argument and will be parsed into conf object
    with the parameter as a variable in uppercase for each parameter/value pair.
    The type parameter is parsed and converted to numpy type string in conf.dataType.

    For example the following is a typical config string for convolution:

        tfloat16-b1-h2-r2-s1-c4-m4-wmin-0.1-wmax0.1-imin1-imax5

    In conf the following variables and values will exist after initialization:
        conf.dataType = "float16"
        conf.B = 1
        conf.H = 2
        conf.R = 2
        conf.S = 1
        conf.C = 4
        conf.M = 4
        conf.WMIN = -0.1
        conf.WMAX = 0.1
        conf.IMIN = 1
        conf.IMAX = 5

argv[2] (optional):
    Specifies the output prefix for generated files. Default is "out_". Variable is conf.outPrefix

argv[3] (optional):
    Specifies the network name. Default is "jdr_v2". Variable is conf.netName

"""

import tensorflow as tf
import numpy as np
import sys
import re
import os
from tensorflow.python.platform import gfile

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

# Extract text-only config options from the config string
# h4-PERM-k1  returns "PERM","h4-k1"
def getPadType(confStr):
  conf = {}
  padType = None
  poolType = None
  usePerm = False
  actType = None
  joinType = None
  for s in ["MAXPOOL", "AVGPOOL"]:
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
  for s in ["RELU", "TANH"]:
    s1 = "-" + s
    if s1 in confStr:
      confStr = confStr.replace(s1, "")
      actType = s
      break
  for s in ["ADD", "SUB"]:
    s1 = "-" + s
    if s1 in confStr:
      confStr = confStr.replace(s1, "")
      joinType = s
      break
  return poolType,padType,actType,joinType,usePerm,confStr

# Trivnet test configurations go into here
class trivnet_conf():
    def __init__(self):
        np.random.seed(17)

        print("\nINFO: started as  ", " ".join(sys.argv))
        dimStr = sys.argv[1]

        # Sample dimStr : b1-h2-r2-s1-c4-m4-wmin-0.1-wmax0.1-imin1-imax5
        dimStr = dimStr.upper() + "-"
        self.fmapValList, dimStr = getConfFmap(dimStr)
        self.poolType, self.padType, self.actType, self.joinType, self.usePerm, dimStr = getPadType(dimStr)

        if len(sys.argv) > 2:
            self.outPrefix = sys.argv[2]
        else:
            self.outPrefix = "out_"

        if len(sys.argv) > 3:
            self.netName = sys.argv[3]
        else:
            self.netName = "jdr_v2"

        dimList = re.split('([A-Z]+)(-?[\d\.]+)-', dimStr)
        params = ["self." + i for i in dimList[1::3]]
        dimCmd = str(tuple(params)).replace("'", "") + " = " + str(tuple(map(float, dimList[2::3]))) 
        dimCmd = dimCmd.replace(".0,", ",")
        dimCmd = dimCmd.replace(".0)", ")")
        print(dimCmd)
        #assert(len(dimList[2::3]) == 1)
        exec(dimCmd)
        if (self.IMIN >= self.IMAX):
            print("Warning: Imin >= Imax:", self.IMIN< self.IMAX)
        if (self.WMIN >= self.WMAX):
            print("Warning: Wmin >= Wmax:", self.WMIN< self.WMAX)

        # DataTypes
        #   npDataType, tfDataType - for the data flow
        dataType = "float16"
        try:
            dataType = "float%d" % self.TFLOAT
        except:
            try:
                dataType = "int%d" % self.TINT
            except:
                try:
                    dataType = "uint%d" % self.TUINT
                except:
                    print("ERROR: no known type, check your t... section of the config string")
                    exit(1)
        for t in ["np", "tf"]:
            exec("self.%sDataType = %s.%s" % (t, t, dataType))

    def gen_array_rand(self, min_val, max_val, shape):      
        tensor = np.random.uniform(min_val, max_val, shape)
        return tensor.astype(self.npDataType)

    def gen_array_linspace(self, min_val, max_val, shape):      
        tensor = np.zeros(shape)
        tensor = permuteArr(np.linspace(min_val, max_val, num=tensor.size, dtype=self.npDataType)).reshape(shape)
        return tensor.astype(self.npDataType)

    def gen_variable_tensor(self, name, initializer):
        return tf.get_variable(
                                name        = name,
                                initializer = initializer,
                                dtype       = self.tfDataType)

    # Generate graph and checkpoint for freezing (freezgin done in Makefile)
    def gen_graph(self, output, input_data, need_freezing=True):
        np.save( self.outPrefix + 'ref_input.npy', input_data)
        print("Inp=\n", input_data)
        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if isinstance(input_data, dict):
                res = sess.run(output, feed_dict=input_data)
            else:                
                res = sess.run(output, feed_dict={"input:0" : input_data})
            print("Res=\n", res)
            print("INFO: the result contains %d infinite numbers" % (res.size - np.count_nonzero(np.isfinite(res))))
            if need_freezing:
                graph = tf.get_default_graph()
                tf.train.write_graph(graph, '.', self.outPrefix + 'graph.pb')
                saver = tf.train.Saver()
                prefixTFfix = ""
                if not self.outPrefix.startswith("/"):
                    prefixTFfix = "./"
                saver.save(sess, prefixTFfix + self.outPrefix + "checkpoint.data")
            else:
                with gfile.GFile(os.path.join('.', self.outPrefix + 'freeze.pb'), "wb") as f:
                    f.write(sess.graph_def.SerializeToString())

conf = trivnet_conf()
