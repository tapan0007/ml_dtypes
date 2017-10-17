# setenv PATH /tmp1/tmp/tensorflow/bin:$PATH

# Nodes etc
#   https://www.tensorflow.org/extend/tool_developers/

# Parsing of pb file
#   https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz

import tensorflow as tf
import numpy as np   
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util
from graphviz import Digraph
import re
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--tfpb', help='TensorFlow freeze graph file', default="f.pb")
parser.add_argument('--out_prefix', help='Prefix for output files', default="out_")
parser.add_argument('--weights', help='Generate weight files', dest="weights",
                    action="store_true",  default=False)
parser.add_argument('--depth', help='Depth of layer name hierarchy to show in the dot output',
                    default=5)

args = parser.parse_args()

file = args.tfpb
if not os.path.isfile(file):
  raise("ERROR: missing --tfpb " + file)


gd = graph_pb2.GraphDef()

dot = Digraph(comment=file)

def tf2dotName(tfName):
  return(re.sub(":\d+$", "", tfName))


with gfile.FastGFile(file,'rb') as f:
  gd.ParseFromString(f.read())
for n in gd.node:
  name = n.name
  print("Name=", name)
  op = n.op
  print("  Op=", op)
  attr = n.attr
  #print("  Attr=", attr)
  t = attr['value'].tensor
  #print("  Tensor=", t.tensor_shape, type(t))
  shapeStr = ""
  try:
    nd = tensor_util.MakeNdarray(t)
    #print("  Nd=", nd.shape)
    shapeStr = " Shape " + "x".join(str(x) for x in nd.shape)
    print("  Nd1=", nd.shape, "ShapeStr=", shapeStr)
  except :
    print("  Nd=Failed")
    shapeStr = ""
  nodeName = tf2dotName(n.name)
  dot.node(nodeName, n.op +
   shapeStr)
  if "conv" in n.op.lower():
    dot.node(nodeName, n.op + shapeStr, color="red")
  
  for ni in n.input:
    print("  Input=", ni)
    dot.edge(tf2dotName(ni), tf2dotName(n.name))
  if (shapeStr):
    weightFile = args.out_prefix + n.name.replace("/", "__")
    print("WeightFile=", weightFile,
          "  Dtype=", nd.dtype,
          "  Size=", nd.size)
    if args.weights:
      np.save(weightFile, nd)

# Add subgraphs
clusters = {}
for n in sorted(gd.node, key=lambda x: x.name):
  clStrs = n.name.split("/")
  c = clusters
  #for i in range(0, len(clStrs)):
  for i in range(0, min(len(clStrs), int(args.depth))):
    if (c.get(clStrs[i]) == None):
      c[clStrs[i]] = {"nodes" : []}
    c = c[clStrs[i]]
  c["nodes"].append(n.name)
print("Clusters=", clusters)


def addClusterNodes(graph, ClusterNode):
  # add nodes in this subgraph
  if ClusterNode.get("nodes") != None:
    for n in ClusterNode["nodes"]:
      print("  DEBUG: added node ", n)
      graph.node(n)
  # add subgraphs
  
  for clShortName in ClusterNode:
    print("  DEBUG: clShortName ", clShortName)
    if clShortName != "nodes":
      with graph.subgraph(name="cluster_" + clShortName) as subGraph:
        subGraph.attr(color="blue")
        subGraph.attr(label=clShortName)
        addClusterNodes(subGraph, ClusterNode.get(clShortName))

if 1:
  addClusterNodes(dot, clusters)


print("Dot=", dot.source)
dot.format = "svg"
dot.render(args.out_prefix + "graph")

