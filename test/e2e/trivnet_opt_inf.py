# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Unit test harness that creates smaller TF pb files from large ones
# The config is path.pb--inpu1,inut2,inputN--output1,output2,...outputN'


import sys, os, re

import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2

# Extract text-only config options from the config string
def getConfOpts(confStr):
  pbFile, inputs, outputs = [None, [], []]
  tokens = confStr.split('--')
  assert len(tokens) == 5
  pbFile = tokens[0]
  inputs = tokens[1].split(',')
  outputs = tokens[2].split(',')
  dtype = tokens[3]
  shape = tokens[4]
  assert len(inputs) > 0
  assert len(outputs) > 0
  return pbFile, inputs, outputs, dtype, shape

def getTfOpForInfPath():
  import tensorflow as tf
  tfPath = sys.modules['tensorflow'].__file__
  optForInf = tfPath.replace('__init__.py', 'python/tools/optimize_for_inference.py')
  assert os.path.isfile(optForInf)
  return optForInf

# Modeled based on $KAENA_EXT_PATH/apps/tf/parallel_wavenet/example1/nsynth_wavenet.patch
def transformGraphPb(inPb, outPb, inputs, outputs, transforms):
  graphDef = graph_pb2.GraphDef()
  with gfile.FastGFile(inPb,'rb') as f:
    graphDef.ParseFromString(f.read())
  graphDefNew = TransformGraph(graphDef, inputs,
                               outputs, transforms)
  tf.train.write_graph(graphDefNew, '.', outPb, as_text=False)
  tf.train.write_graph(graphDefNew, '.', outPb + ".txt", as_text=True)
  
  # Test it
  sess = tf.Session()
  tf.import_graph_def(graphDefNew, name="")
  with sess.as_default() as sess:
    graph = sess.graph

  print('INFO: wrote ', outPb)
  

print("\nINFO: started as  ", " ".join(sys.argv))

confStr, outPrefix, netName = sys.argv[1:4]

kePath = os.environ.get('KAENA_EXT_PATH')
pbFile, inputs, outputs, dtype, shape = getConfOpts(confStr)
print("INFO: options config ", pbFile, inputs, outputs, dtype, shape)

'''
optForInf = getTfOpForInfPath()
cmd = 'python3 %s --frozen_graph --input %s/%s/%s --output %s ' % (optForInf, kePath, 'apps/tf', pbFile, 'trivnet_freeze.pb')
cmd += '--input_names %s '  % ','.join(inputs)
cmd += '--output_names %s ' % ','.join(outputs)

print('INFO: executing', cmd)
os.system(cmd)
'''

transforms = [
  'strip_unused_nodes(type=%s, shape="%s")' % (dtype, shape),
  ]
pbIn = '%s/%s/%s' % (kePath, 'apps/tf', pbFile)
pbOut = 'trivnet_freeze.pb'
transformGraphPb(pbIn, pbOut, inputs, outputs, transforms)
