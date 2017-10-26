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
from PIL import Image
import csv

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
    
    # Iterate over all TF graph definition nodes
    for tfNode in self.__gd.node:
      tfop = TfOp(tfNode.name, tfNode.op, tfNode)
      #print(tfop)
      if (re.search(focusNodeRe, tfNode.name) != None):
        self.__kg.addNode(tfNode.name, {"tfop" : tfop, 'op_type' : tfNode.op})
        numOps += 1
        if (re.search("conv", tfop.op, re.I) != None):
          numConv += 1
      
        for ni in tfNode.input:
          #print("  Input=", ni)
          # Nodes out of focus may not exist, so skip the edge too
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


  def writeImages(self, outPrefix, imageFile):
    inputNode = self.__kg.getNode("input")
    self.__kg.levelize()
    inputTfOpName = inputNode.getAttr("tfop").name
    with tf.Session() as sess:
      tf.import_graph_def(self.__gd, name="")
      graph = sess.graph
      inputOp = graph.get_operation_by_name(inputTfOpName)
      inputTensor = inputOp.outputs[0]
      inputShape = inputTensor.get_shape().as_list()
      shapeXY = inputShape[1:3]
      img = Image.open(imageFile).resize(shapeXY)
      img = np.array(img)
      img = img.reshape(inputShape)

      tfVars = []
      kNodes = []
      
      # Perform serialization of operations by using levels
      levelizedNodes = self.__kg.getLevelizedNodes()
      for level in range(0, len(levelizedNodes)):
        for n in levelizedNodes[level]:
          #print("DEBUG: node=", n.getName())
          tfOpName = n.getAttr("tfop").name
          op = graph.get_operation_by_name(tfOpName)
          tensor = op.outputs[0]
          shape = tensor.get_shape().as_list()
          n.setAttr("np_shape", shape)
          tfVars.append(tensor.name)
          kNodes.append(n)
          # Add support for multi-output graphs
          assert(len(op.outputs)==1)
      
      print("INFO: identified %d operations, computing ..." % len(tfVars))
      numImages = 0
      tfResults = sess.run(tfVars, feed_dict={inputTensor.name : img})
      perDot = max(1, int(len(tfVars) / 80))
      print("INFO: writing if/ofmap files ...")
      for (n, var, nd) in zip(kNodes, tfVars, tfResults):
        if nd.size > 0:
          imageFile = outPrefix + var.replace("/", "__")
          #print("ImageFile=", weightFile,
          #      "  Dtype=", nd.dtype,
          #      "  Size=", nd.size)
          np.save(imageFile, nd)
          numImages += 1
          if numImages % perDot == 0:
            print(".", end='', flush=True)
          n.setAttr("np_file", imageFile + ".npy")
          dtype = nd.dtype
          n.setAttr("np_dtype", str(dtype))
        else:
          print("INFO: Failed to get tensor content for ",
                tfOp.type, "  ", tfOp.name)
      print("")
    print("INFO: wrote %d i/ofmap files" % numImages)


  # Writes graph in a given format using graphviz lib
  # Key operations like convolution are colored red
  # The depth determines subgraph clastering based on operation manes
  #   layerN/branchM/convP cluster would be breated (in blue) if depth is 3
  def writeDot(self, depth, outFile, outFormat = "svg"):
    dot = Digraph(comment="writeDot")
    for n in self.__kg.getNodes():
      tfOp = n.getAttr("tfop")
      attrs = {}
      if re.search("conv", tfOp.op, re.I):
        attrs["color"] = "red"
      dot.node(n.getName(), tfOp.op, attrs)

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
      if "nodes" in ClusterNode:
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
    
  
  @staticmethod
  def tf2dotName(tfName):
    return(re.sub(":\d+$", "", tfName))

  @staticmethod
  def npShapeToSize(shapeList):
    arr = np.ndarray(shapeList)
    return(arr.size)


  def writeOpsCsv(self, csvFile):
    levelizedNodes = self.__kg.getLevelizedNodes()
    with open(csvFile, 'w') as csvHandle:
      fieldNames = ["OpName", "OpType", "Level",
                    "OutputData", "OutputSize", "InputSize",
                    "OutputShape", "OutputFile"]
      for i in range(0,6):
        fieldNames += ["Input" + str(i) + "Shape",
                       "Input" + str(i) + "File"]
      writer = csv.DictWriter(csvHandle, fieldnames=fieldNames)

      rows = []
      for level in range(0, len(levelizedNodes)):
        for n in levelizedNodes[level]:
          #print("DEBUG: node=", n.getName())
          (opName, opType, dType, npShape, npFile) = n.getNpInfo()
          row = {"OpName"      : opName,
                 "OpType"      : opType,
                 "Level"       : level,
                 "OutputData"  : dType, 
                 "OutputSize"  : TfFe.npShapeToSize(npShape), 
                 "OutputShape" : npShape, 
                 "OutputFile"  : npFile}
          
          i = 0
          inputSize = 0
          for preNode in self.__kg.nodePredecessors(n):
            (opName, opType, dType, npShape, npFile) = preNode.getNpInfo()
            row["Input" + str(i) + "Shape"] = npShape
            row["Input" + str(i) + "File"] = npFile
            inputSize += TfFe.npShapeToSize(npShape)
            i += 1
          row["InputSize"] = inputSize
          rows.append(row)
      writer.writeheader()
      writer.writerows(rows)
    print("INFO: Wrote op sequences into " + csvFile)



