# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Kaena Compiler front end for TensorFlow 

# Nodes etc
#   https://www.tensorflow.org/extend/tool_developers/

# Parsing of pb file - sample pb
#   https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz

import tensorflow as tf
import numpy as np
from keras.applications.resnet50 import preprocess_input
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import tag_constants
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
  def __init__(self, dataPathWidthThreshold, debugLevel, dotTimeout, batch, showOpNameInKgraph, inputNamesToFormat):
    self.__gd = None
    self.__kg = None
    self.dataPathWidthThreshold = dataPathWidthThreshold  # Min tensor size for visualization
    self.debugLevel = debugLevel
    kog.Config.Dot.timeout = dotTimeout
    kog.Config.Graph.showOpNameInKgraph = showOpNameInKgraph
    kog.Config.Graph.inputNamesToFormat = inputNamesToFormat
    self.kaenaPath = os.environ["KAENA_PATH"]
    self.batch = batch
  
  def getKaenaOpGraph(self):
    return(self.__kg)
  
  def loadPb(self, pbFile, focusNodeRe, focusToName):
    self.sess = None
    if os.path.isdir(pbFile):
      # Load model checkpoint
      graph = tf.Graph()
      with graph.as_default():
        # Grow GPU memory as needed at the cost of fragmentation.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.08
        sess = tf.Session(graph=graph, config=config)
        tf.saved_model.loader.load( sess, [tag_constants.SERVING], pbFile)
        self.__gd = graph.as_graph_def()
        self.sess = sess
    elif pbFile.endswith(".pbtxt"):
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
    
    # Simplistic abstration of GraphDephs for --focus_to.
    # It uses strings (not node pointers) to store basic graph connectivity
    class tfGraph(object):
      def __init__(self, tfGd, debugLevel):
        self.debugLevel = debugLevel
        #self.name2node = {}
        #for tfNode in tfGd.node:
        #  self.name2node[tfNode.name] = tfNode
        self.name2predNameList = {}
        for tfNode in tfGd.node:
          predNames = []
          if self.debugLevel >= 2:
            print("  DEBUG tfGraph node", tfNode.name)
          for ni in tfNode.input:
            if self.debugLevel >= 2:
              print("    DEBUG tfGraph node input", ni)
            m = re.findall("(.*):(\d+)$", ni)
            if len(m) == 1:
              (fromName, fromIndex) = m[0][0], int(m[0][1])
            else:
              (fromName, fromIndex) = (ni, 0)
            predNames.append(fromName)
          self.name2predNameList[tfNode.name] = predNames
          if self.debugLevel >= 2:
            print("  DEBUG tfGraph node", tfNode.name, "added predecessors", predNames)
      # Returns set of node names
      # All APIs are string based, not nodes (for simplicity & debug)
      # Do not use for general graph operations due to efficiency
      def faninConeNodes(self, toNode):
        faninCone = set([toNode])
        newFaninCone = set()
        fLen = 0
        while len(faninCone) > fLen:
          newFaninCone = faninCone
          fLen = len(faninCone)
          for nodeName in faninCone:
            newFaninCone = newFaninCone.union(set(self.name2predNameList[nodeName]))
            #print(len(fanin), len(newFanin))
          faninCone = newFaninCone
        return faninCone
    
    if focusToName != None:
      tfg = tfGraph(self.__gd, self.debugLevel)
      focusToNameSet = tfg.faninConeNodes(focusToName)
      if self.debugLevel >= 1:
        print("INFO: --focus_to %s resulted in nodes:" % focusToName)
        for nodeName in sorted(focusToNameSet):
          print("  ", nodeName)
    else:
      focusToNameSet = None
    
    # Add all nodes (ops) in TF graph definition
    for tfNode in self.__gd.node:
      tfop = TfOp(tfNode.name, tfNode.op, tfNode)
      if self.debugLevel >= 3:
        print("\nDEBUG loadPb ", tfop, tfNode)
      if (re.search(focusNodeRe, tfNode.name) != None) and (
           focusToNameSet == None or tfNode.name in focusToNameSet):
      
        add_attrs = {}
        #for attr in ["data_format"]:
        #  if attr in tfNode.attr:
        #    add_attrs[attr] = tfNode.attr[attr]
        #    print("  DEBUG attr=", attr, "  ", add_attrs[attr])        
        add_attrs["tfop"] = tfop
        numOps += 1
        node = None
        if (re.search("Conv2DBackpropInput", tfop.op, re.I) != None):
          numConv += 1
          node = kog.NodeConv2DTranspose(tfNode.name, tfop.op, add_attrs)
        elif (re.search("conv", tfop.op, re.I) != None):
          numConv += 1
          node = kog.NodeConv2D(tfNode.name, tfop.op, add_attrs)
          #print("DEBUG created NodeConv2D")
        elif (re.search("Sub|Add|BiasAdd|ExpandDims", tfop.op, re.I) != None):
          node = kog.NodeSimple2(tfNode.name, tfop.op, add_attrs)
          #print("DEBUG created NodeSimple2")
        elif (re.search("Softmax", tfop.op, re.I) != None):
          node = kog.NodeSoftmax(tfNode.name, tfop.op, add_attrs)
        elif (re.search("MatMul", tfop.op, re.I) != None):
          node = kog.NodeMatMul(tfNode.name, tfop.op, add_attrs)
        elif (re.search("Reshape", tfop.op, re.I) != None):
          node = kog.NodeReshape(tfNode.name, tfop.op, add_attrs)
        elif  (re.search("relu|lrelu|tanh|Softplus|Sigmoid|Softmax", tfop.op, re.I) != None):
          node = kog.NodeSimple(tfNode.name, tfop.op, add_attrs)
        elif (re.search("Squeeze|Exp", tfop.op, re.I) != None):
          node = kog.NodeSimple(tfNode.name, tfop.op, add_attrs)
          #print("DEBUG created NodeSimple")
        elif (re.search("MaxPool|AvgPool", tfop.op, re.I) != None):
          node = kog.NodePool(tfNode.name, tfop.op, add_attrs)
          #print("DEBUG created NodeMaxPool")
        #elif (re.search("Const", tfop.op, re.I) != None):
        #  node = kog.NodeConst(tfNode.name, tfop.op, add_attrs)
        #  #print("DEBUG created NodeConst")
        elif (re.search("Placeholder", tfop.op, re.I) != None):
        #print("DEBUG created NodeInput")
          node = kog.NodeInput(tfNode.name, tfop.op, add_attrs)
        elif (re.search("StridedSlice", tfop.op, re.I) != None):
          node = kog.NodeStridedSlice(tfNode.name, tfop.op, add_attrs)
        elif (re.search("^Unstack$|^Unpack$", tfop.op, re.I) != None):
          node = kog.NodeUnstack(tfNode.name, "Unstack", add_attrs)
        elif (re.search("^Split$", tfop.op, re.I) != None):
          node = kog.NodeSplit(tfNode.name, "Split", add_attrs)
        elif (re.search("^Multiply$|^Mul$", tfop.op, re.I) != None):
          node = kog.NodeMultiply(tfNode.name, "Multiply", add_attrs)
        elif (re.search("^Maximum$|^Minimum$", tfop.op, re.I) != None):
          node = kog.NodeMultiply(tfNode.name, tfop.op, add_attrs)
        elif (re.search("Concat", tfop.op, re.I) != None):
          node = kog.NodeConcat(tfNode.name, "Concat", add_attrs)   # Concat, ConcatV2
        elif (re.search("ClipByValue", tfop.op, re.I) != None):
          node = kog.NodeClipByValue(tfNode.name, tfop.op, add_attrs)
        elif (re.search("Slice", tfop.op, re.I) != None):
          node = kog.NodeSlice(tfNode.name, tfop.op, add_attrs)
        elif (re.search("Pad", tfop.op, re.I) != None):
          node = kog.NodePad(tfNode.name, tfop.op, add_attrs)
        elif (re.search("Transpose", tfop.op, re.I) != None):
          node = kog.NodeTranspose(tfNode.name, tfop.op, add_attrs)
        elif (re.search("SpaceToBatch|BatchToSpace", tfop.op, re.I) != None):
          node = kog.NodeSpaceBatch(tfNode.name, tfop.op, add_attrs)
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
        np.save(weightFile, np.ascontiguousarray(nd))
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
  
  def writeImages(self, outPrefix, imageFiles, inputNodeNames, inputConstants, excludeOpsFromCaptureRe,
                  preprocessor, preprocessor_args):
    self.__kg.levelize()
    inputNodes = []
    if len(inputNodeNames) == 0:
      # Auto detect input - take 1st placeholder
      inputNodes = [x for x in self.__kg.getNodes() if x.getOpType() == "Placeholder"]
    else:
      for inputName in inputNodeNames:
        if self.__kg.hasNode(inputName):
          inputNodes.append(self.__kg.getNode(inputName))
        else:
          lowestLevelNodes = self.__kg.getLowestLevelNodes()
          print("ERROR: the  --input_node %s  was not found. Use one of  %s" % (inputNode, [ n.getName() for n in lowestLevelNodes]))
          exit(1)
    assert len(inputNodes) > 0
    self.__kg.setInputNodes(inputNodes)
    
    inputFeedDict = self.inputConst2dict(inputConstants)
    inputOverrideDict = {}
    
    # Grow GPU memory as needed at the cost of fragmentation.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.08
    if self.sess == None:
      self.sess = tf.Session(config=config)
      tf.import_graph_def(self.__gd, name="")
    with self.sess.as_default() as sess:
      graph = sess.graph
      
      # Batched proprocessor uses individual images. TO_DO: need multi-input preoprocessor UI
      if preprocessor == "" and len(inputNodes) != len(imageFiles):
        raise ValueError("ERROR: provide %d items in --images" % len(inputNodes))
      for i in range(len(inputNodes)):
        inputNode = inputNodes[i]
        imageFile = imageFiles[i]
        print("INFO: generating input image %d for node %s  using %s" % (i, inputNode.getName(), imageFile)) 
       
        inputTfOpName = inputNode.getAttr("tfop").name
        inputOp = graph.get_operation_by_name(inputTfOpName)
        inputTensor = inputOp.outputs[0]
        inputShape = inputTensor.get_shape().as_list()
        shapeXY = inputShape[1:3]
        inputType = inputTensor.dtype.as_numpy_dtype

        # Handle resnet50 input shape without batching, e.g.,[None, 32, 32, 3]
        if len(inputShape) > 0:
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
          import tempfile
          with tempfile.NamedTemporaryFile(suffix='.npy') as ntf:
            tfName = ntf.name
            cmd = "{cmd} {args} --inputs {images} --output {output}".format(cmd=preprocessor, args=preprocessor_args,
                                                                            images=' '.join(imageFiles), output=tfName)
            print("INFO: Running preprocessor:: ", cmd)
            os.system(cmd)
            img = np.load(tfName)

        img = img.reshape(inputShape)
        inputOverrideDict.update({inputTensor.name : img.copy()})
        print("INFO: generated input image for node %s of shape %s and type %s" % (inputNode.getName(), img.shape, img.dtype)) 
        

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
                # FIX_THIS: some transform graph results loose shape so
                #   we we may need to add 2nd pass of tensor capture (those with none shape)
                #if tensor.name == "resnet_v2_152/block1/unit_1/bottleneck_v2/preact/Relu:0":
                #  print("DEBUG")
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
          # num_split is for Split operator 
          for attr in ["strides", "padding", "data_format", "ksize", "num_split"]:
            if attr in op.node_def.attr:
              n.setAttr(attr, op.get_attr(attr))
              #print("  DEBUG attr=", attr, "  ", op.get_attr(attr))          
          # LSTM attributes - keeping separate from the above loop during debug phase
          for attr in ["begin_mask", "ellipsis_mask", "end_mask", "new_axis_mask", "shrink_axis_mask", "axis"]:
            if attr in op.node_def.attr:
              n.setAttr(attr, op.get_attr(attr))
              #print("  DEBUG attr=", attr, "  ", op.get_attr(attr))          
          
      
      print("INFO: identified %d tensors, computing ..." % len(tfVars))
      if self.debugLevel > 0:
        print("DEBUG: tensor names: %s" % tfVars)
      numImages = 0
      inputFeedDict.update(inputOverrideDict)
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
        if self.debugLevel > 0:
           print("DEBUG: captured tensor %s of size %d" % (var, nd.size))
        if nd.size > 0:
          imageFile = outPrefix + var.replace("/", "__")
          #print("ImageFile=", weightFile,
          #      "  Dtype=", nd.dtype,
          #      "  Size=", nd.size)
          np.save(imageFile, np.ascontiguousarray(nd))
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
  

