# TensorFlow front end for Kaena Compiler

# Nodes etc
#   https://www.tensorflow.org/extend/tool_developers/

# Parsing of pb file - sample pb
#   https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz

import tensorflow as tf
import numpy as np   
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util
from graphviz import Digraph
import re
import KaenaOpGraph as kog

class TfOp:
  def __init__(self, name, op, tfNode):
    self.name = name
    self.op = op
    self.tfNode = tfNode
  def __str__(self):
    return("Name=" + self.name + "  Op=" + self.op)
  def getTensor(self):
    attr = self.tfNode.attr
    #print("  Attr=", attr)
    tensor = attr['value'].tensor
    return(tensor)
  def getTensorNd(self):
    tensor = self.getTensor()
    #print("  Tensor=", tfTensor.tensor_shape, type(t))
    shapeStr = ""
    try:
      nd = tensor_util.MakeNdarray(tensor)
      #print("  Nd=", nd.shape)
      shapeStr = " Shape " + "x".join(str(x) for x in nd.shape)
      #print("  Nd1=", nd.shape, "ShapeStr=", shapeStr)
    except :
      #print("  Nd=Failed")
      shapeStr = ""
      nd = np.array([])
    return(nd)

class TfFe:
  def __init__(self):
    self.__gd = None
    self.__kg = None
  
  def loadPb(self, pbFile, focusNodeRe):
    self.__gd = graph_pb2.GraphDef()
    with gfile.FastGFile(pbFile,'rb') as f:
      self.__gd.ParseFromString(f.read())
    
    self.__kg = kog.Graph(pbFile)
    numOps = 0
    numConv = 0
    
    for tfNode in self.__gd.node:
      tfop = TfOp(tfNode.name, tfNode.op, tfNode)
      #print(tfop)
      if (re.search(focusNodeRe, tfNode.name) != None):
        self.__kg.addNode(tfNode.name, {"tfop" : tfop})
        numOps += 1
        if (re.search("conv", tfop.op.lower(), re.I) != None):
          numConv += 1
      
        for ni in tfNode.input:
          #print("  Input=", ni)
          # Nodes out of focus may not edist, so skip the edge to
          if (self.__kg.hasNode(ni) and self.__kg.hasNode(tfNode.name)):
            self.__kg.addEdge(ni, tfNode.name)
    print("INFO: loaded %s file with %d ops  of which %d are CONV"
          % (pbFile, numOps, numConv))
    
  def writeWeights(self, outPrefix):
    numWeights = 0
    for n in self.__kg.getNodes():
      tfOp = n.getAttr("tfop")
      #tfNode = tfOp.tfNode
      nd = tfOp.getTensorNd()
      if (nd.size > 0):
        weightFile = outPrefix + tfOp.name.replace("/", "__")
        #print("WeightFile=", weightFile,
        #      "  Dtype=", nd.dtype,
        #      "  Size=", nd.size)
        np.save(weightFile, nd)
        numWeights += 1
    print("INFO: wrote %d weights" % numWeights)

  def writeDot(self, depth, outFile, outFormat = "svg"):
    dot = Digraph(comment="writeDot")
    for n in self.__kg.getNodes():
      tfOp = n.getAttr("tfop")
      dot.node(n.getName(), tfOp.op)
    for e in self.__kg.getEdges():
      #print(e)
      dot.edge(e.fromNode().getName(), e.toNode().getName())

    # Add subgraphs
    clusters = {}
    for n in sorted(self.__kg.getNodes(), key=lambda x: x.getName()):
      clStrs = n.getName().split("/")
      c = clusters
      #for i in range(0, len(clStrs)):
      for i in range(0, min(len(clStrs), depth)):
        if (c.get(clStrs[i]) == None):
          c[clStrs[i]] = {"nodes" : []}
        c = c[clStrs[i]]
      c["nodes"].append(n.getName())
    #print("Clusters=", clusters)

    def addClusterNodes(graph, ClusterNode):
      # add nodes in this subgraph
      if ClusterNode.get("nodes") != None:
        for n in ClusterNode["nodes"]:
          #print("  DEBUG: added node ", n)
          graph.node(n)
      # add subgraphs

      for clShortName in ClusterNode:
        #print("  DEBUG: clShortName ", clShortName)
        if clShortName != "nodes":
          with graph.subgraph(name="cluster_" + clShortName) as subGraph:
            subGraph.attr(color="blue")
            subGraph.attr(label=clShortName)
            addClusterNodes(subGraph, ClusterNode.get(clShortName))

    if 1:
      addClusterNodes(dot, clusters)

    #print("Dot=", dot.source)
    dot.format = outFormat
    dot.render(outFile)
    print("INFO: wrote " + outFile + "." + outFormat)
    
  
  def tf2dotName(tfName):
    return(re.sub(":\d+$", "", tfName))

