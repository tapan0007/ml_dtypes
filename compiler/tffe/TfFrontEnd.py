# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Kaena Compiler front end for TensorFlow 

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
  def __init__(self, dataPathWidthThreshold):
    self.__gd = None
    self.__kg = None
    self.dataPathWidthThreshold = dataPathWidthThreshold  # Min tensor size for visualization
  
  def getKaenaOpGraph(self):
    return(self.__kg)
  
  def loadPb(self, pbFile, focusNodeRe):
    self.__gd = graph_pb2.GraphDef()
    with gfile.FastGFile(pbFile,'rb') as f:
      self.__gd.ParseFromString(f.read())
    
    self.__kg = kog.Graph(pbFile)
    numOps = 0
    numConv = 0
    
    # Add all nodes (ops) in TF graph definition
    for tfNode in self.__gd.node:
      tfop = TfOp(tfNode.name, tfNode.op, tfNode)
      #print("\nDEBUG loadPb ", tfop, tfNode)
      if (re.search(focusNodeRe, tfNode.name) != None):
      
        add_attrs = {}
        for attr in ["padding", "data_format"]:
          if attr in tfNode.attr:
            add_attrs[attr] = tfNode.attr[attr]
            #print("  DEBUG attr=", attr, "  ", add_attrs[attr])        
        add_attrs["tfop"] = tfop
        numOps += 1
        node = None
        if (re.search("conv", tfop.op, re.I) != None):
          numConv += 1
          #print("DEBUG strides=", tfNode.attr["strides"])
          #print("DEBUG padding=", tfNode.attr["padding"])
          #print("DEBUG data_format=", tfNode.attr["data_format"])
          node = kog.NodeConv2D(tfNode.name, tfop.op, add_attrs["padding"], add_attrs["data_format"], add_attrs)
        else:
          node = kog.Node(tfNode.name, tfop.op, add_attrs)
        self.__kg.addNode(node)
    print("INFO: loaded %s file with %d ops  of which %d are CONV"
          % (pbFile, numOps, numConv))

    # Add all edges (ops) in TF graph definition
    for tfNode in self.__gd.node:
      tfop = TfOp(tfNode.name, tfNode.op, tfNode)
      #print("DEBUG: tfop.op=", tfop.op)
      if (re.search(focusNodeRe, tfNode.name) != None):
        i = 0
        for ni in tfNode.input:
          #print("  Input=", ni)
          fromIndex = 0
          m = re.findall("(.*):(\d+)$", ni)
          if len(m) == 1:
            (fromName, fromIndex) = m[0][0], int(m[0][1])
          else:
            (fromName, fromIndex) = (ni, 0)
          # Nodes out of focus may not exist, so skip the edge too
          if (self.__kg.hasNode(fromName) and self.__kg.hasNode(tfNode.name)):
            self.__kg.addEdge(fromName, fromIndex, tfNode.name, i)
          else:
            print("INFO: skipped edge %s -> %s due to missing nodes"
                  % (ni, tfNode.name))
          i += 1
    
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


  def writeImages(self, outPrefix, imageFile, inputTensorName):
    inputNode = self.__kg.getNode(inputTensorName)
    assert(inputNode != None)
    self.__kg.setInputNode(inputNode)
    self.__kg.levelize()
    inputTfOpName = inputNode.getAttr("tfop").name
    with tf.Session() as sess:
      tf.import_graph_def(self.__gd, name="")
      graph = sess.graph
      inputOp = graph.get_operation_by_name(inputTfOpName)
      inputTensor = inputOp.outputs[0]
      inputShape = inputTensor.get_shape().as_list()
      shapeXY = inputShape[1:3]
      if imageFile.endswith(".npy"):
        img = np.load(imageFile)
      elif imageFile == "linear":
        unusedArr = np.ndarray(inputShape)
        img = np.arange(unusedArr.size, dtype=np.float16).reshape(inputShape)
        print("INFO: generated linear input=\n", img)
      elif " " in imageFile:
        img = np.fromstring(imageFile, dtype=np.float16, sep=" ")
      else:
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
          tfOpName = n.getOpName()
          op = graph.get_operation_by_name(tfOpName)
          #if (re.search("conv", n.getOpType(), re.I) != None):
            #print("DEBUG: conv  ", n.getOpName())
          for tensor in op.outputs:
            shape = tensor.get_shape().as_list()
            npInfo = kog.NpInfo(tensor.name, shape)
            n.appendNpInfo(npInfo)
            tfVars.append(tensor.name)
            kNodes.append((n, npInfo))
          # update/collect attributes
          # Strides are in the pb but require complex parsing (op.get_attr)
          #   which seems only accessible from the graph so deferred to calibration
          for attr in ["strides"]:
            if attr in op.node_def.attr:
              n.setAttr(attr, op.get_attr(attr))
              print("  DEBUG attr=", attr, "  ", op.get_attr(attr))    
          
      
      print("INFO: identified %d tensors, computing ..." % len(tfVars))
      numImages = 0
      tfResults = sess.run(tfVars, feed_dict={inputTensor.name : img})
      perDot = max(1, int(len(tfVars) / 80))
      print("INFO: writing if/ofmap files ...")
      for ((n, npInfo), var, nd) in zip(kNodes, tfVars, tfResults):
        if nd.size > 0:
          imageFile = outPrefix + var.replace("/", "__")
          #print("ImageFile=", weightFile,
          #      "  Dtype=", nd.dtype,
          #      "  Size=", nd.size)
          np.save(imageFile, nd)
          numImages += 1
          if numImages % perDot == 0:
            print(".", end='', flush=True)
          npInfo.npFile = imageFile + ".npy"
          npInfo.dType = str(nd.dtype)
        else:
          print("INFO: Failed to get tensor content for ",
                tfOp.type, "  ", tfOp.name)
      print("")
    print("INFO: wrote %d i/ofmap files" % numImages)


  # Writes graph in a given format using graphviz lib
  # Key operations like convolution are colored red
  # The depth determines subgraph clastering based on operation names
  #   layerN/branchM/convP cluster would be breated (in blue) if depth is 3
  def writeDot(self, depth, outFile, outFormat = "svg"):
    dot = Digraph(comment="writeDot")
    for n in self.__kg.getNodes():
      tfOp = n.getAttr("tfop")
      attrs = {}
      if re.search("conv", tfOp.op, re.I):
        attrs["color"] = "red"
      dot.node(n.getName(), tfOp.op, attrs)

    for edge in self.__kg.getEdges():
      #print(edge)
      #print("DEBUG: adding edge to dot ", edge)
      #print("DEBUG: attrs=", edge.getAttrs())
      dot.edge(edge.getFromPosNode().node.getName(),
               edge.getToPosNode().node.getName(),
               edge.getLabel(), edge.getAttrs())

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

  # Color graph by the datatype. Intended for int8 inteference so 8b is green
  @staticmethod
  def dType2color(dType):
    return {
      "uint8"   : "black:green",
      "int8"    : "black:green",
      "int32"   : "black:yellow",
      "float16" : "black:pink",
      "float32" : "black:red",
      "float64" : "black:purple"
    }.get(dType, None)

  # Generates csv report for all inputs and outputs of the operations
  # Also populates the Kaena graph with tensor (edge) size labels,
  # data flow colors (eventually these 2 functions could be split at
  # the cost of replicating the loops and data querry)
  def writeOpsCsv(self, csvFile):
    levelizedNodes = self.__kg.getLevelizedNodes()
    
    # Count number of inputs and outputs
    numInputs = 0
    numOutputs = 0
    for level in range(0, len(levelizedNodes)):
      for n in levelizedNodes[level]:
        numInputs = max(numInputs, len(n.getFaninEdges()))
        numOutputs = max(numOutputs, len(n.getFanoutEdges()))

    rows = []
    #debugId = 0
    for level in range(0, len(levelizedNodes)):
      for n in levelizedNodes[level]:
        #print("DEBUG: node=", n.getName())
        opName = n.getOpName()
        opType = n.getOpType()
        #if (re.search("conv", opType, re.I) != None):
        #  print("DEBUG: conv  ", opName)
        row = {"OpName"      : opName,
               "OpType"      : opType,
               "Level"       : level}
        npOutputInfo = n.getNpInfo()
        outputSize = 0
        for i in range(0, len(npOutputInfo)):
          npInfo = npOutputInfo[i]
          row["Output" + str(i) + "dType"] = npInfo.dType
          row["Output" + str(i) + "Shape"] = npInfo.npShape
          row["Output" + str(i) + "File"]  = npInfo.npFile
          outputSize += TfFe.npShapeToSize(npInfo.npShape)
        row["OutputSize"] = outputSize
        inputSize = 0
        faninEdges = n.getFaninEdges()
        for i in range(0, len(faninEdges)):
          edge = faninEdges[i]
          p = edge.getFromPosNode()
          (fromNode, fromIndex) = (p.node, p.index)
          npInfo = fromNode.getNpInfo()[fromIndex]
          row["Input" + str(i) + "dType"] = npInfo.dType
          row["Input" + str(i) + "Shape"] = npInfo.npShape
          row["Input" + str(i) + "File"] = npInfo.npFile
          inputSize += TfFe.npShapeToSize(npInfo.npShape)
          
          # Populate edge attributes for plotting type and size
          edge.setLabel(str(npInfo.dType) + "\\n" + str(npInfo.npShape))
          if self.__kg.edgeIsInMainFlow(edge):
            if TfFe.npShapeToSize(npInfo.npShape) >= self.dataPathWidthThreshold:
              #edge.setAttr("penwidth", str(5 + debugId / 1000))
              edge.setAttr("penwidth", str(5))
              edge.setAttr("weight", str(2))
              color = TfFe.dType2color(npInfo.dType)
              if color:
                edge.setAttr("color", color)
                #print("DEBUG: set color ", color, " on edge ", edge, "  DEBUG_ID=", debugId)
          i += 1
          #edge.setAttr("DENUG_ID", str(debugId))
          #debugId += 1
        #print("\n")
        if 0:
          for i in range(0, len(faninEdges)):
            edge = faninEdges[i]
            #print("DEBUG: final csv edge attributes ", edge.getAttrs(), " Edge=", edge)
        row["InputSize"] = inputSize
        rows.append(row)
        #print("DEBUG: row=", row)
    
    # Write csv
    with open(csvFile, 'w') as csvHandle:
      fieldNames = ["OpName", "OpType", "Level",
                    "OutputSize", "InputSize"]
      for i in range(0,numOutputs):
        fieldNames += ["Output" + str(i) + "dType",
                       "Output" + str(i) + "Shape",
                       "Output" + str(i) + "File"]
      for i in range(0,numInputs):
        fieldNames += ["Input" + str(i) + "dType",
                       "Input" + str(i) + "Shape",
                       "Input" + str(i) + "File"]
      writer = csv.DictWriter(csvHandle, fieldnames=fieldNames)
      writer.writeheader()
      writer.writerows(rows)
    print("INFO: Wrote op sequences into " + csvFile)



