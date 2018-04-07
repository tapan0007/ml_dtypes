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
from google.protobuf import text_format
from graphviz import Digraph
import re
import KaenaOpGraph as kog
from PIL import Image
import csv

import sys
import os
sys.path.insert(0, os.environ["KAENA_PATH"] + "/compiler/tffe")
import MiscUtil

class TfOp:
  def __init__(self, name, op, tfNode):
    self.name = name
    self.op = op
    self.tfNode = tfNode
    self.shape = self.extractProtoShape(tfNode)
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
  def extractProtoShape(self, tfNode):
    shape = []
    try:
      attrVal = tfNode.attr.get('value', None)
      if attrVal:
        dimList = attrVal.tensor.tensor_shape.dim
        for dim in dimList:
          val = re.sub(r'[^\d]', "", str(dim))
          shape.append(int(val))
    except:
      pass
    if len(shape) == 0:
      try:
        attrVal = tfNode.attr.get('_output_shapes', None)
        if attrVal:
          dimStr = tfNode.attr['_output_shapes'].list.shape[0]
          dimStr1 = re.sub(r'[^\d]', ' ', str(dimStr))
          shape = dimStr1.split()
          shape = [int(d) for d in shape]
      except:
        pass
    if len(shape) == 0:
      try:
        attrVal = tfNode.attr.get('shape', None)
        if attrVal:
          dimStr = tfNode.attr['shape'].shape
          dimStr1 = re.sub(r'[^\d]', ' ', str(dimStr))
          shape = dimStr1.split()
          shape = [int(d) for d in shape]
      except:
        pass
    return shape

class TfFe:
  def __init__(self, dataPathWidthThreshold, debugLevel, dotTimeout, batch):
    self.__gd = None
    self.__kg = None
    self.dataPathWidthThreshold = dataPathWidthThreshold  # Min tensor size for visualization
    self.debugLevel = debugLevel
    kog.Config.Dot.timeout = dotTimeout
    self.kaenaPath = os.environ["KAENA_PATH"]
    self.batch = batch
  
  def getKaenaOpGraph(self):
    return(self.__kg)
  
  def loadPb(self, pbFile, focusNodeRe):
    if pbFile.endswith(".pbtxt"):
      self.__gd = graph_pb2.GraphDef()
      with open(pbFile) as f:
        self.__gd = text_format.Parse(f.read(), tf.GraphDef())
    else:
      self.__gd = graph_pb2.GraphDef()
      with gfile.FastGFile(pbFile,'rb') as f:
        self.__gd.ParseFromString(f.read())

    self.__kg = kog.Graph(debugLevel=self.debugLevel)
    kog.Config.debugLevel = self.debugLevel
    numOps = 0
    numConv = 0
    
    # Add all nodes (ops) in TF graph definition
    for tfNode in self.__gd.node:
      tfop = TfOp(tfNode.name, tfNode.op, tfNode)
      if self.debugLevel >= 3:
        print("\nDEBUG loadPb ", tfop, tfNode)
      if (re.search(focusNodeRe, tfNode.name) != None):
      
        add_attrs = {}
        #for attr in ["data_format"]:
        #  if attr in tfNode.attr:
        #    add_attrs[attr] = tfNode.attr[attr]
        #    print("  DEBUG attr=", attr, "  ", add_attrs[attr])        
        add_attrs["tfop"] = tfop
        numOps += 1
        node = None
        if (re.search("conv", tfop.op, re.I) != None):
          numConv += 1
          node = kog.NodeConv2D(tfNode.name, tfop.op, add_attrs)
          #print("DEBUG created NodeConv2D")
        elif (re.search("Add|BiasAdd", tfop.op, re.I) != None):
          node = kog.NodeSimple2(tfNode.name, tfop.op, add_attrs)
          #print("DEBUG created NodeSimple2")
        elif (re.search("Softmax", tfop.op, re.I) != None):
          node = kog.NodeSoftmax(tfNode.name, tfop.op, add_attrs)
        elif (re.search("MatMul", tfop.op, re.I) != None):
          node = kog.NodeMatMul(tfNode.name, tfop.op, add_attrs)
        elif (re.search("Reshape", tfop.op, re.I) != None):
          node = kog.NodeReshape(tfNode.name, tfop.op, add_attrs)
        elif  (re.search("relu|tanh|Sigmoid|Softmax", tfop.op, re.I) != None):
          node = kog.NodeSimple(tfNode.name, tfop.op, add_attrs)
          #print("DEBUG created NodeSimple")
        elif (re.search("MaxPool|AvgPool", tfop.op, re.I) != None):
          node = kog.NodePool(tfNode.name, tfop.op, add_attrs)
          #print("DEBUG created NodeMaxPool")
        #elif (re.search("Const", tfop.op, re.I) != None):
        #  node = kog.NodeConst(tfNode.name, tfop.op, add_attrs)
        #  #print("DEBUG created NodeConst")
        elif (re.search("Placeholder", tfop.op, re.I) != None and
              re.search("input", tfop.name, re.I) != None):
        #print("DEBUG created NodeInput")
          node = kog.NodeInput(tfNode.name, tfop.op, add_attrs)
        elif (re.search("StridedSlice", tfop.op, re.I) != None):
          node = kog.NodeStridedSlice(tfNode.name, tfop.op, add_attrs)
        elif (re.search("^Unstack$|^Unpack$", tfop.op, re.I) != None):
          node = kog.NodeUnstack(tfNode.name, "Unstack", add_attrs)
        elif (re.search("^Multiply$|^Mul$", tfop.op, re.I) != None):
          node = kog.NodeMultiply(tfNode.name, "Multiply", add_attrs)
        else:
          node = kog.Node(tfNode.name, tfop.op, add_attrs)
        node.setProtoShape(tfop.shape)
        self.__kg.addNode(node)
        #print ("DEBUG: loadpb: adding node %s, type %s" % (node.getName(), type(node)))
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

  # Convert --input_constants variable valueOrNpy pairs to feed dictionary
  def inputConst2dict(self, inputConstants):
    feedDict = {}
    if len(inputConstants) > 0:
      for var, valStr in zip(inputConstants[0::2], inputConstants[1::2]):
        val = None
        try:
          val = eval(valStr)
        except:
          val = np.load(valStr)
        feedDict[var] = val
    return feedDict
  
  def writeImages(self, outPrefix, imageFile, inputNodeName, inputConstants, excludeOpsFromCaptureRe):
    self.__kg.levelize()
    inputNodes = []
    if inputNodeName == None:
      # Auto detect input - take 1st placeholder
      inputNodes = [x for x in self.__kg.getNodes() if x.getOpType() == "Placeholder"]
    elif self.__kg.hasNode(inputNodeName):
      inputNodes = [self.__kg.getNode(inputNodeName)]
    else:
      lowestLevelNodes = self.__kg.getLowestLevelNodes()
      print("ERROR: the  --input_node %s  was not found. Use one of  %s" % (inputNodeName, [ n.getName() for n in lowestLevelNodes]))
      exit(1)
    assert len(inputNodes) > 0
    self.__kg.setInputNodes(inputNodes)
    inputTfOpName = inputNodes[0].getAttr("tfop").name
    
    inputFeedDict = self.inputConst2dict(inputConstants)
    
    # Grow GPU memory as needed at the cost of fragmentation.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.08
    with tf.Session(config=config) as sess:
      tf.import_graph_def(self.__gd, name="")
      graph = sess.graph
      inputOp = graph.get_operation_by_name(inputTfOpName)
      inputTensor = inputOp.outputs[0]
      inputShape = inputTensor.get_shape().as_list()
      shapeXY = inputShape[1:3]
      inputType = inputTensor.dtype.as_numpy_dtype()
      
      # Handle resnet50 input shape without batching, e.g.,[None, 32, 32, 3]
      if inputShape[0] == None:
        inputShape[0] = self.batch
      
      if imageFile.endswith(".npy"):
        img = np.load(imageFile)
      elif imageFile == "linear":
        unusedArr = np.ndarray(inputShape)
        img = np.arange(unusedArr.size, dtype=inputType).reshape(inputShape)
        print("INFO: generated linear input=\n", img)
      elif imageFile == "linspace1":
        unusedArr = np.ndarray(inputShape)
        img = np.linspace(0, 1, num=unusedArr.size, dtype=inputType).reshape(inputShape)
        print("INFO: generated linear input=\n", img)
      elif " " in imageFile:
        img = np.fromstring(imageFile, dtype=inputType, sep=" ")
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
          if (re.search("StridedSlice", n.getOpType(), re.I) != None):
            print("DEBUG: StridedSlice  ", n.getOpName())
          if excludeOpsFromCaptureRe == None or not re.match(excludeOpsFromCaptureRe, tfOpName):
            for tensor in op.outputs:
              shapeRaw = tensor.get_shape()
              if shapeRaw == None:
                print("INFO: Skipping capture on node with shape None %s" % tensor.name)
                continue
              shape = shapeRaw.as_list()
              # Handle missing batch dimension on some input nodes
              if len(shape) > 0 and shape[0] == None:
                shape[0] = 1
                print("WARNING: adjusted batch dimension \"None\" to 1 on  %s  %s  %s" %
                  (n.getOpType(), n.getName(), str(shape))) 
              if len(shape) == 0:
                print("WARNING: zero-dimension op output on  %s  %s  %s" %
                  (n.getOpType(), n.getName(), str(shape))) 
              else:
                # Generic case excluding batching above
                for i in range(len(shape)):
                  if shape[i] == None:
                    shape[i] = 1
                    print("WARNING: dimension %d \"None\" to 1 on  %s  %s  %s" %
                      (i, n.getOpType(), n.getName(), str(shape))) 
              npInfo = kog.NpInfo(tensor.name, shape)
              #lookedUpTensor = graph.get_tensor_by_name(tensor.name)
              #if lookedUpTensor == None:
              #  print("DEBUG: tensor %s not found" % tensor.name)
              n.appendNpInfo(npInfo)
              tfVars.append(tensor.name)
              kNodes.append((n, npInfo))
          else:
            print("INFO: Excluded node from capture %s" % tfOpName)
          # update/collect attributes
          # Strides are in the pb but require complex parsing (op.get_attr)
          #   which seems only accessible from the graph so deferred to calibration
          for attr in ["strides", "padding", "data_format", "ksize"]:
            if attr in op.node_def.attr:
              n.setAttr(attr, op.get_attr(attr))
              #print("  DEBUG attr=", attr, "  ", op.get_attr(attr))          
          # LSTM attributes - keeping separate from the above loop during debug phase
          for attr in ["begin_mask", "ellipsis_mask", "end_mask", "new_axis_mask", "shrink_axis_mask", "axis"]:
            if attr in op.node_def.attr:
              n.setAttr(attr, op.get_attr(attr))
              #print("  DEBUG attr=", attr, "  ", op.get_attr(attr))          
          
      
      print("INFO: identified %d tensors, computing ..." % len(tfVars))
      numImages = 0
      inputFeedDict.update({inputTensor.name : img})
      #tfVars = ['dropout_1/keras_learning_phase:0']
      
      # For --exclude_ops_from_capture
      diagnoseCompiledOutVars = False
      if diagnoseCompiledOutVars:
        for tmpVar in tfVars:
          try:
            tfResults = sess.run(tmpVar, feed_dict=inputFeedDict)
          except:
            print("DEBUG: TF Failed to compute %s" % tmpVar)
      else:
        tfResults = sess.run(tfVars, feed_dict=inputFeedDict)

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
          # Update shapes that TF binds late, e.g., batch
          updatedShape = list(nd.shape)
          if updatedShape != npInfo.npShape:
            print("\nINFO: updated tensort shape from %s to %s on  %s  %s" %
                   (str(npInfo.npShape), str(updatedShape), n.getOpType(), n.getName()))
            npInfo.npShape = updatedShape
        else:
          print("INFO: Failed to get tensor content for ",
                var, "  ", n.getOpName(), n.getOpType())
      print("")
    print("INFO: wrote %d i/ofmap files" % numImages)


  @staticmethod
  def tf2dotName(tfName):
    return(re.sub(":\d+$", "", tfName))

  @staticmethod
  def npShapeToSize(shapeList):
    arr = np.ndarray(shapeList)
    return arr.size

  # Color graph by the datatype. Intended for int8 inteference so 8b is green
  @staticmethod
  def dType2color(dType):
    #print("DEBUG: dType=%s" % str(dType))
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
        npOutputInfo = n.getNpInfo()
        numInputs = max(numInputs, len(n.getFaninEdges()))
        numOutputs = max(numOutputs, len(npOutputInfo))

    rows = []
    #debugId = 0
    for level in range(0, len(levelizedNodes)):
      for n in levelizedNodes[level]:
        #print("DEBUG: node=", n.getName())
        opName = n.getOpName()
        opType = n.getOpType()
        opArgsText = n.getOpArgsText()
        #if (re.search("conv", opType, re.I) != None):
        #  print("DEBUG: conv  ", opName)
        row = {"OpName"      : opName,
               "OpType"      : opType,
               "OpArgs"      : opArgsText,
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
          fromNpInfos = fromNode.getNpInfo()
          if len(fromNpInfos) > fromIndex:
            npInfo = fromNpInfos[fromIndex]
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
      fieldNames = ["OpName", "OpType", "OpArgs", "Level",
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
  

