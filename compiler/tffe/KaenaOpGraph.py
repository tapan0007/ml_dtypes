# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Kaena abstraction of neural network framework operations
# Suitable for TF freeze graph input and general op reduction and fusing

from NpTransforms import NpTrans as npt
from NpTransforms import TensorFormat, TensorFormatMap
from NpUtils import NpUtils as npu
import os, re, json, sys
import numpy as np
import math
import random
import datetime
from graphviz import Digraph
from collections import OrderedDict
from functools import total_ordering

sys.path.insert(0, os.environ["KAENA_PATH"] + "/compiler/tffe")
import MiscUtil

def getInfoDateStr():
  return "INFO %s :" % str(datetime.datetime.now())

class Config:
  debugLevel = 0
  class Tpb:
    numTpb = 1
    specTops = 32
    freq = 1e9
    poolGiBps = 64*2 * freq / 2**30  # 64 columns of float16 per 1 GHz cycle in Giga bytes
      # AWS terminology is G = 1e9, Gi 2**30
  class Ddr:
    utilization = 0.9
    GiBps = 42 * utilization
  class Pe:
    minWave = 64
    maxWave = 256
  class Graph:
    legendText = """
  Conv2D, MaxPool, Add  ... Operators
  Strides, Kernel  ... Arguments
  w0.125 i0.191 o0.766 MiB  ... weight, input, output
                               tensor sizes in MegaBytes
  OpWB 784 ... operations per byte of weights
  BT(n)    ... batch targets for n TPBs
  1-2-5    ... recommended batches for roofline-minWave-maxWave
               Batch 0 means tiling required
"""
    showOpNameInKgraph = False
    inputNamesToFormat = {}
    useWCFormat = False
    useHWCFormat = False
  class Dot:
    timeout = 60
  class Scheduler:
    waveoptOptions = ""

class Object:
  def __init__(self, name, attrs):
    self.__name = name
    self.__attrs = attrs.copy()
  def getName(self):
    return(self.__name)
  def getAttr(self, attrName):
    return self.__attrs.get(attrName, None)
  def getAttrs(self):
    return(self.__attrs)
  def setAttr(self, attrName, attrVal):
    self.__attrs[attrName] = attrVal

# Neural network Numpy-style info for each op output
class NpInfo:
  def __init__(self, tensorName, npShape):
    self.tensorName = tensorName
    self.npShape = npShape
    self.npFile = None
    self.dType = None
  def nbytes(self):
    return np.empty(self.npShape, dtype=self.dType).nbytes
  def getValues(self):
    arr = np.load(self.npFile)
    return arr

# NN operation
@total_ordering
class Node(Object):
  def __init__(self, name, opType, attrs):
    Object.__init__(self, name, attrs)
    self.__level = -1
    self.__opType = opType
    self.__npInfo = []
    self.__fanin = []  # inputs
    self.__fanout = [] # outputs
    self.protoShape = []
  def copy(self):
    CL = type(self)
    n = CL(self.getName(), self.getOpType(), self.getAttrs())
    n.__npInfo = self.getNpInfo()
    #print("DEBUG: copy node class %s  %s %s" % (CL,  self.getOpType(), self.getName()))
    return n
  def copyAs(self, CL, opType):
    n = CL(self.getName(), opType, self.getAttrs())
    n.__npInfo = self.getNpInfo()
    #print("DEBUG: copy node class %s  %s %s" % (CL,  self.getOpType(), self.getName()))
    return n
  def __str__(self):
    return("Node=" + self.getName())
  def __lt__(self, other):
    return self.getName() <= other.getName()
  def getLevel(self):
    return(self.__level)
  def setLevel(self, level):
    self.__level = level
  def getNpInfo(self):
    return(self.__npInfo)
  def getInputNodesAndNpInfo(self):
    inputNodesAndNpInfos = []
    faninEdges = self.getFaninEdges()
    for i in range(0, len(faninEdges)):
      edge = faninEdges[i]
      p = edge.getFromPosNode()
      (fromNode, fromIndex) = (p.node, p.index)
      npInfo = fromNode.getNpInfo()[fromIndex]
      inputNodesAndNpInfos.append((fromNode, npInfo))
    return(inputNodesAndNpInfos)
  def getOpType(self):
    return(self.__opType)
  def getOpName(self):
    return(self.getName())
  def appendNpInfo(self, npInfo):
    self.__npInfo.append(npInfo)

  def disconnectFaninEdge(self, edge):
    assert(edge in self.__fanin)
    for i in range(len(self.__fanin)):
        if edge == self.__fanin[i]:
            self.__fanin[i] = None
            break
    assert(not edge in self.__fanin)

  def disconnectFanoutEdge(self, outPinIdx, edge):
    assert(outPinIdx < len(self.__fanout))
    edgesOfOutput = self.__fanout[outPinIdx]
    assert(edge in edgesOfOutput)
    edgesOfOutput.remove(edge)

  def getFaninEdges(self):
    return(self.__fanin)
  def getFaninMainFlowEdges(self):
    return [e for e in self.getFaninEdges() if e.isInMainFlow()]
  def getFanoutEdges(self):
    return([item for edgelist in self.__fanout for item in edgelist])

  def getFanouts(self):
    return self.__fanout

  def getFanoutEdgesOfOutput(self, outIdx):
    return [edge for edge in self.__fanout[outIdx]]

  def getFanoutMainFlowEdges(self):
    return [e for e in self.getFanoutEdges() if e.isInMainFlow()]
  def getFanoutMainFlowNodes(self):
    return [e.getToNode() for e in self.getFanoutMainFlowEdges()]
  def getFaninMainFlowNodes(self):
    return [e.getFromNode() for e in self.getFaninMainFlowEdges()]
  def hasMainFlowPredecessorsInSet(self, nodeSet):
    predNodes = set(self.getFaninMainFlowNodes())
    return predNodes.issubset(nodeSet)
  # Like graph class nodeSuccesors, but a) localized to Node class, 2) return list of [position, node name]
  def getFanoutNodePosNames(self):
    nodeList = []
    for pos in range(len(self.__fanout)):
      edgeList = self.__fanout[pos]
      for edge in edgeList:
        toNode = edge.getToNode()
        nodeList.append([pos, toNode.getName()])
    return nodeList
  # Edge between 2 nodes (from this to another)
  def getEdgeTo(self, toNode):
    for e in self.getFanoutEdges():
      p = e.getToPosNode()
      if p.node == toNode:
        return e
    return None
  # Fanin of 1 per input
  def setFaninEdge(self, edge, index):
    assert(len(self.__fanin) < index + 1 or self.__fanin[index] == None)
    while len(self.__fanin) < index + 1:
      self.__fanin.append(None)
    self.__fanin[index] = edge
  # Fanout can be greater than 1
  def setFanoutEdge(self, edge, index):
    while len(self.__fanout) < index + 1:
      self.__fanout.append([])
    self.__fanout[index].append(edge)

  #def addFanoutEdge(self, edge, outIdx):
  #  if not self.__fanout:
  #      self.__fanout = []
  #  while len(self.__fanout) < outIdx + 1:
  #    self.__fanout.append([])
  #  self.__fanout[outIdx].append(edge)

  #def addFaninEdge(self, edge):
  #  if not self.__fanin:
  #      self.__fanin = []
  #  assert(not edge in self.__fanin)
  #  self.__fanin.append(edge)

  # Base class support for single-input, single output layers
  # E.g., activation, later possibly other simple layers
  # Returns list of layer maps (e.g., 2 for a node with a side constant),
  # and list of npy files (that need to be part of Kgraph package)
  def genCompilerLayerJson(self, tensorFormatMap):
    return([{"layer_type" :  self.getOpType(),
            "layer_name" :  self.getOpName(),
            "out_data_type": self.getNpInfo()[0].dType,
            "#comment"   :  "Unsupported operation"
            }], [])

  # Helper for op counts - number of scalar elements in all output tensors
  def getNpyInfoSize(self):
    size = 0;
    npInfos = self.getNpInfo()
    # Count 1 op per each output's tensor scalar
    for i in range(len(npInfos)):
      size += np.empty(npInfos[i].npShape).size
    return size

  # Size in bytes in all output tensors
  def getNpyInfoBytes(self):
    numBytes = 0;
    npInfos = self.getNpInfo()
    for npInfo in npInfos:
      numBytes += npInfo.nbytes()
    return numBytes

  def getTpbShape(self, npInfo):
    if len(npInfo.npShape) == 3:
      if Config.Graph.useHWCFormat:
        tfShape4D = npt.hwcShapeToNHWC(npInfo.npShape)
      else:
        tfShape4D = npt.nwcShapeToNHWC(npInfo.npShape)
    elif len(npInfo.npShape) == 2:
      if Config.Graph.useWCFormat:
        tfShape4D = npt.wcShapeToNHWC(npInfo.npShape)
      else:
        tfShape4D = npt.ncShapeToNHWC(npInfo.npShape)
    elif len(npInfo.npShape) == 1:
      tfShape4D = npt.cShapeToNHWC(npInfo.npShape)
    elif len(npInfo.npShape) == 0:
      tfShape4D = []
    else:
      assert len(npInfo.npShape) == 4
      tfShape4D = npInfo.npShape
    tpbShape = list(npt.reorderShape(tfShape4D, npt.TF, npt.SIM, npt.Fmaps))      
    return tpbShape


  def convertShapeForConstWeights(self, npInfoIF1, tensorFormatMap):
    # The matrix side input is handled like convolution weights
    tfShape4Dw = npt.cmShapeToRSCM(npInfoIF1.npShape)
    (npFileSimW, simFormatW)  = npt.copyNpyFileAs(npInfoIF1.npFile, npt.TF, npt.SIM, npt.Weights, tfShape4Dw)
    tensorFormatMap.add(npInfoIF1.tensorName,
                        TensorFormat(npInfoIF1.tensorName, self.getOpName(),
                                     npInfoIF1.npFile, npt.CM,
                                     npFileSimW, simFormatW, True))  # const in CNNs, may need to split the node class for LSTMs
    tpbShape4Dw = list(npt.reorderShape(tfShape4Dw, npt.TF, npt.SIM, npt.Weights))
    return (tpbShape4Dw, simFormatW, npFileSimW)                        

  def convertShape(self, npInfo, tensorFormatMap, isConst=False):

    if len(npInfo.npShape) == 3:
      if Config.Graph.useHWCFormat:
        tfShape4D = npt.hwcShapeToNHWC(npInfo.npShape)
        tfFormat = npt.HWC  
      else:
        tfShape4D = npt.nwcShapeToNHWC(npInfo.npShape)
        tfFormat = npt.NWC
    elif len(npInfo.npShape) == 2:
      if Config.Graph.useWCFormat:
        tfShape4D = npt.wcShapeToNHWC(npInfo.npShape)
        tfFormat = npt.WC        
      else:
        tfShape4D = npt.ncShapeToNHWC(npInfo.npShape)
        tfFormat = npt.NC
    elif len(npInfo.npShape) == 1:
      tfShape4D = npt.cShapeToNHWC(npInfo.npShape)
      tfFormat = npt.C
    elif len(npInfo.npShape) == 0:
      tfShape4D = []
      tfFormat = ""
    else:
      assert len(npInfo.npShape) == 4
      tfShape4D = npInfo.npShape
      tfFormat = npt.Formats[npt.TF][npt.Fmaps]

    # Overwrite format from command line
    simFormat = ""
    if self.getName() in Config.Graph.inputNamesToFormat and not isConst:
      # overriding is limited to Fmaps now.
      tfFormat = Config.Graph.inputNamesToFormat[self.getName()]
      assert(len(tfFormat) == len(npInfo.npShape))
      if tfFormat == npt.NWC:
        tfShape4D = npt.nwcShapeToNHWC(npInfo.npShape)
      elif tfFormat == npt.HNC:
        tfShape4D = npt.hncShapeToHNWC(npInfo.npShape)
        simFormat = npt.HNWC
        (npFileSim, tpbShape) = npt.formatNpyFileAs(npInfo.npFile, npt.HNC, simFormat)
      elif tfFormat == npt.NC:
        tfShape4D = npt.ncShapeToNHWC(npInfo.npShape)
      elif tfFormat == npt.NW:
        tfShape4D = npt.nwShapeToNHWC(npInfo.npShape)
      elif tfFormat == npt.C:
        tfShape4D = npt.cShapeToNHWC(npInfo.npShape)
      elif tfFormat == npt.Formats[npt.TF][npt.Fmaps]:
        tfShape4D = npInfo.npShape
      else:
        print("ERROR: input format %s is not recognized (must be one of C, NC, NWC, HNC, or NHWC)"%self.inputFormat)

    if simFormat == "":
        if tfShape4D != []:
            tpbShape = list(npt.reorderShape(tfShape4D, npt.TF, npt.SIM, npt.Fmaps))
            (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D)
        else:
            tpbShape = []
            (npFileSim, simFormat) = (npInfo.npFile, tfFormat)
    tensorFormatMap.add(npInfo.tensorName,
                        TensorFormat(npInfo.tensorName, self.getOpName(),
                                     npInfo.npFile, tfFormat,
                                     npFileSim, simFormat, isConst))
    return (tpbShape, simFormat, npFileSim)                        
       
  def convertDimParams(self, dimParams, insertVal=1):

    if len(dimParams) == 3:
      if Config.Graph.useHWCFormat:
        tfShape4D = npt.hwcShapeToNHWC(dimParams, insertVal)
        tfFormat = npt.HWC  
      else:
        tfShape4D = npt.nwcShapeToNHWC(dimParams, insertVal)
        tfFormat = npt.NWC
    elif len(dimParams) == 2:
      if Config.Graph.useWCFormat:
        tfShape4D = npt.wcShapeToNHWC(dimParams, insertVal)
        tfFormat = npt.WC        
      else:
        tfShape4D = npt.ncShapeToNHWC(dimParams, insertVal)
        tfFormat = npt.NC
    elif len(dimParams) == 1:
      tfShape4D = npt.cShapeToNHWC(dimParams, insertVal)
      tfFormat = npt.C
    elif len(dimParams) == 0:
      tfShape4D = []
      tfFormat = ""
    else:
      assert len(dimParams) == 4
      tfShape4D = dimParams
      tfFormat = npt.Formats[npt.TF][npt.Fmaps]

    # Overwrite format from command line
    simFormat = ""
    if self.getName() in Config.Graph.inputNamesToFormat:
      # overriding is limited to Fmaps now.
      tfFormat = Config.Graph.inputNamesToFormat[self.getName()]
      assert(len(tfFormat) == len(dimParams))
      if tfFormat == npt.NWC:
        tfShape4D = npt.nwcShapeToNHWC(dimParams, insertVal)
      elif tfFormat == npt.HNC:
        tfShape4D = npt.hncShapeToHNWC(dimParams, insertVal)
      elif tfFormat == npt.NC:
        tfShape4D = npt.ncShapeToNHWC(dimParams, insertVal)
      elif tfFormat == npt.NW:
        tfShape4D = npt.nwShapeToNHWC(dimParams, insertVal)
      elif tfFormat == npt.C:
        tfShape4D = npt.cShapeToNHWC(dimParams, insertVal)
      elif tfFormat == npt.Formats[npt.TF][npt.Fmaps]:
        tfShape4D = dimParams
      else:
        print("ERROR: input format %s is not recognized (must be one of C, NC, NWC, HNC, or NHWC)"%self.inputFormat)

    if simFormat == "":
        if tfShape4D != []:
            tpbShape = list(npt.reorderShape(tfShape4D, npt.TF, npt.SIM, npt.Fmaps))
        else:
            tpbShape = []
    return tpbShape

  # Describes computational complexity of the operation
  def getOpCount(self, padded=False):
    opCount = 1
    npInfos = self.getNpInfo()
    if len(npInfos) > 0:
      opCount += self.getNpyInfoSize()
    return opCount
  def getDotText(self):
    text = self.getOpType()
    if Config.Graph.showOpNameInKgraph:
      text += "\n" + self.getName()
    text += "\n" + str(self.protoShape)
    return text
  # Supported ops/nodes are passed down through the compiler and simulator flow
  def isSupported(self):
    return False
  def isConst(self):
    return False
  def getOpArgsText(self):
    argsText = self.getDotText()
    # Simple implementation - reuse dot text and remove \n
    argsText = re.sub("\n", " ", argsText)
    return argsText
  def isMainFlowNode(self):
    return len(self.getFaninMainFlowEdges()) > 0 or len(self.getFanoutMainFlowEdges()) > 0
  def setProtoShape(self, shape):
    self.protoShape = shape
  def getWeightBytes(self):
    return 0

class PosNode:
  def __init__(self, node, index):
    self.node = node
    self.index = index

  def getName(self):
    return self.node.getName() + ":" + str(self.index)

class Edge(Object):
  def __init__(self, fromPosNode, toPosNode, attrs):
    Object.__init__(self, "Edge", attrs)
    self.__fromPosNode = fromPosNode
    self.__toPosNode = toPosNode
    self.__label = None
    self.__isInMainFlow = False
  def getFromPosNode(self):
    return(self.__fromPosNode)
  def getToPosNode(self):
    return(self.__toPosNode)
  def getFromNode(self):
    return self.getFromPosNode().node
  def getToNode(self):
    return self.getToPosNode().node
  def setLabel(self, label):
    self.__label = label
  def getLabel(self):
    return(self.__label)
  def __str__(self):
    f = self.getFromPosNode()
    t = self.getToPosNode()
    return("Edge" +
           "  From=" + f.node.getName() + ":" + str(f.index) +
           "  To=" + t.node.getName()  + ":" + str(t.index) )
  def setIsInMainFlow(self, isInMainFlow):
    self.__isInMainFlow = isInMainFlow
  def isInMainFlow(self):
    return self.__isInMainFlow


###############################################################################
# Constant node - single output node
###############################################################################
class NodeConst(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)
    # FIX_THIS: shorterm hack to get transformer working, tracked by kaena-1126
    if self.getFanoutEdges(): # to prevent the case of a dangling const node
      toNode = self.getFanoutEdges()[0].getToNode()
      if toNode.getOpType() == 'MatMul' and toNode.getFaninEdges()[0].getFromNode() is self:
        tpbShape, simFormat, npFileSim = self.convertShape(npInfo, tensorFormatMap, self.isConst())

    # Spec for future global format tracking
    #  (newShape, newFile) = npTt.translate("NC", npt.FmapsSIM, npt.FmapsopName, npInfo.npShape, npInfo.npFile)
    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "out_data_type"   : npInfo.dType,
      "ref_file"        : npFileSim,
      "previous_layers" : [],
      "#comment"        : "Constants such as weights or biases"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

  def isConst(self):
    return True

###############################################################################
# Simple single input, single output nodes like RELU
###############################################################################
class NodeSimple(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)

    # FIX_THIS - IFMAP, it should not be needed
    in_nodes = self.getInputNodesAndNpInfo()
    fromIfNode = in_nodes[0][0]
    #((fromIfNode, npInfoIF),) = self.getInputNodesAndNpInfo()
    #(npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [fromIfNode.getName()],
      "#comment"        : "Simple operation with single input tensor and single output tensor"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

###############################################################################
# BatchToSpace and SpaceToBatch
###############################################################################
class NodeSpaceBatch(NodeSimple):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    (layerDataBase, fileListBase) = super().genCompilerLayerJson(tensorFormatMap)
    layerData = layerDataBase[0]
    in_nodes = self.getInputNodesAndNpInfo()
    # Extract block size
    blk_shape_tmp = in_nodes[1][1].getValues()
    #assert(blk_shape_tmp.shape == 2)
    blk_shape = [np.asscalar(j) for j in blk_shape_tmp]
    # Extract padding/crop values
    values_tmp = in_nodes[2][1].getValues()
    assert(values_tmp.shape == (2,2))
    values  = [[np.asscalar(values_tmp[i][j])
                for j in range(values_tmp.shape[1])]
                for i in range(values_tmp.shape[0])]
    # Convert values to NCHW
    values_NCHW = [[0,0], [0,0], [0,0], [0,0]]
    values_NCHW[2] = values[0]
    values_NCHW[3] = values[1]
    blk_shape_NCHW = [1, 1, 1, 1]
    blk_shape_NCHW[2] = blk_shape[0]
    blk_shape_NCHW[3] = blk_shape[1]
    layerData["block_shape"] = blk_shape_NCHW
    if re.search("SpaceToBatch", layerData["layer_type"]):
        layerData["#comment"] = "Zero pad spatial dimensions and deinterleave spatial data (deinterleave dimension is merged with batch dimension)"
        layerData["padding"] = values_NCHW
    else:
        layerData["#comment"] = "Interleave spatial data (reverse of SpaceToBatch) followed by cropping"
        layerData["crop"] = values_NCHW
    return(layerDataBase, fileListBase)

###############################################################################
# A node for Concat operation (Multi inputs and single output)
###############################################################################
class NodeConcat(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)

    # FIX_THIS - IFMAP, it should not be needed
    #((fromIfNode, npInfoIF),) = self.getInputNodesAndNpInfo()
    fromIfNode = []
    npInfoIf = []
    for m in range (0, len(self.getInputNodesAndNpInfo()) - 1):
      fromIfNode.append(self.getInputNodesAndNpInfo()[m][0].getName())
      npInfoIf.append(self.getInputNodesAndNpInfo()[m][1])
    #(npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
    layerData = {
#      "axis"            :
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : fromIfNode,
      "#comment"        : "Concatenate along channel dimension, using order in previous_layers list"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

###############################################################################
# Softmax
###############################################################################
class NodeSoftmax(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)

    ((fromIfNode, _),) = self.getInputNodesAndNpInfo()
    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [fromIfNode.getName()],
      "#comment"        : "Softmax operation"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    #assert(layerDataBase[0]["ofmap_format"] == "NHWC")
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

  def getOpCount(self, padded=False):
    # Softmax computes, exp, sum, and divide so 3x
    opCount = 3 * self.getNpyInfoSize()
    return opCount


###############################################################################
# Input Node
###############################################################################
class NodeInput(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]

    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)

    # TODO: if assertion fires, this sequence of subgraphs cannot execute on tonga (without help from transpose code on cpu).
    #assert npInfo.getValues().tobytes() == np.load(npFileSim).tobytes()

    layerData = {
      "layer_name"      : self.getName(),
      "layer_type"      : "Input",
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [],
      "#comment"        : "Input placeholder"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

###############################################################################
# Base class for nodes that use striding and padding
#   Example: what is common between some ops; in tf.nn syntax:
#    conv2d(
#     C   input,
#        filter,
#     C   strides,
#     C   padding,
#        use_cudnn_on_gpu=True,
#     C   data_format='NHWC',
#        dilations=[1, 1, 1, 1],
#     C   name=None
#    )
#    max_pool(
#     C   value,
#        ksize,
#     C   strides,
#     C   padding,
#     C   data_format='NHWC',
#     C   name=None
#    )
#    avg_pool(
#     C   value,
#        ksize,
#     C   strides,
#     C   padding,
#     C   data_format='NHWC',
#     C   name=None
#    )
###############################################################################
class NodeBasePaddedStrided(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  def getStrides(self):
    return self.getAttr("strides")

  def getPaddingMode(self):
    return self.getAttr("padding").decode("utf-8")

  def getDataFormat(self):
    return self.getAttr("data_format").decode("utf-8")

  def isSupported(self):
    return True

  # Utility to extract 2-D object dimensions from 4-D (batched, with channels) conv tensor dimension
  def dim2imgSize(self, dim):
    assert len(dim) == 4
    assert self.getDataFormat() == "NHWC"
    arr = dim[1:3]
    assert dim[0] > 0
    assert dim[3] > 0
    return arr

  # Return the 2 significant dimensions of batched multi channel IFMAP (of 2-D images)
  def getIFmapImageSize(self):
    (fromIfNode, npInfoIF) = self.getInputNodesAndNpInfo()[0]
    ifmapArr = self.dim2imgSize(npInfoIF.npShape)
    return ifmapArr

  # Return the 2 significant dimensions of batched multi channel OFMAP (of 2-D images)
  def getOFmapImageSize(self):
    npInfo = self.getNpInfo()[0]
    ifmapArr = self.dim2imgSize(npInfo.npShape)
    return ifmapArr

  # Return the 2 significant dimensions of batched multi channel strides (on 2-D images)
  def getStridesSize(self):
    strides = self.getStrides()
    stridesArr = self.dim2imgSize(strides)
    return stridesArr

  # Helper function to calculate padding in SAME mode given
  # stride, filter, image sizes
  @staticmethod
  def calcSamePadding(S, R, H):
    Ho = (H + S - 1) // S
    spacing = S - R  # negative spacing means overlap
    inPixels = Ho * R + (Ho - 1) * spacing
    leftover = max(0, inPixels - H)
    return(Ho, leftover // 2, (leftover + 1) // 2)

  def calcTpbPadding(self, kernelShape2D, paddingMode):
    padding = None
    (R, S) = kernelShape2D
    (Hi, Wi) = self.getIFmapImageSize()
    (Ho, Wo) = self.getOFmapImageSize()
    (Sv, Sh) = self.getStridesSize()
    if paddingMode == "SAME":
      # Basic case with stride 1
      #   Filter   IFMAP        Tonga padding
      #          0123456789 -> OFMAP pixel
      #   1      0        9     [0, 0]
      #   2      01       9P    [0, 1]
      #   3     P01      89P    [1, 1]
      #   4     P012     89PP   [1, 2]
      #   5    PP012    789PP   [2, 2]
      #   6    PP0123   789PPP  [2, 3]
      #   7   PPP0123  6789PPP  [3, 3]
      #
      # Stride 2 and up :
      #   Unified formula is based on uniform spacing (or overlap):
      #    Filter Spacing Filter ... Spacing Filter
      if Ho <= Hi:  # conv
        (HoCalc, padNorth, padSouth) = self.calcSamePadding(Sv, R, Hi)
        assert Ho == HoCalc
      else:  # conv transpose
        (HiCalc, padNorth, padSouth) = self.calcSamePadding(Sv, R, Ho)
        assert Hi == HiCalc
      if Wo <= Wi:  # conv
        (WoCalc, padWest,  padEast)  = self.calcSamePadding(Sh, S, Wi)
        assert Wo == WoCalc
      else:   # conv transpose
        (WiCalc, padWest,  padEast)  = self.calcSamePadding(Sh, S, Wo)
        assert Wi == WiCalc
    elif paddingMode == "VALID":
      # Valid mode should not have any padding so just assert on proper Fmap sizes
      #   assert Ho * Sv + R // 2 * 2 == Hi
      # However in 16b resent there is a conv2d with image 55, filter 1,
      #   stride 2, output image 28 in which case the above does not hold
      assert((Hi - (R-Sv)) // Sv == Ho)
      (padNorth, padSouth) = (0, 0)
      (padWest, padEast) = (0, 0)
    else:
      raise("Unsupported padding mode %s" % paddingMode)
    if Config.debugLevel >= 1:
      print("DEBUG: calcTpbPadding  %s  IFMAP %dx%d  OFMAP %dx%d  STRIDE %dx%d  FILTER %dx%d  MODE %s  PAD %d-%dx%d-%d" %
       (self.getName(), Hi, Wi, Ho, Wo, Sv, Sh, R, S, paddingMode, padNorth,padSouth, padWest, padEast))
    padding = [[0,0], [0,0], [padNorth,padSouth], [padWest,padEast]]
    return padding

###############################################################################
# 2D convolution
###############################################################################
class NodeConv2D(NodeBasePaddedStrided):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Return the height and width dimensions of 2-D filter
  def getFilter2D(self):
    ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW)) = self.getInputNodesAndNpInfo()
    filterArr = npt.subShape(npInfoW.npShape, "RS", npt.TF, npt.Weights)
    return filterArr

  # Return the height and width dimensions of 2-D feature map
  def getImg2D(self):
    npInfo = self.getNpInfo()[0]
    img2D = npt.subShape(npInfo.npShape, "HW", npt.TF, npt.Fmaps)
    return img2D

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW)) = self.getInputNodesAndNpInfo()
    
    # OFMAP
    (npFileSim, simFormatOF) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    tensorFormatMap.add(npInfo.tensorName,
                        TensorFormat(npInfo.tensorName, self.getOpName(),
                                     npInfo.npFile, npt.Formats[npt.TF][npt.Fmaps],
                                     npFileSim, simFormatOF, False))
    
    # IFMAP, not needed
    (npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
    tensorFormatMap.add(npInfoIF.tensorName,
                        TensorFormat(npInfoIF.tensorName, self.getOpName(),
                                     npInfoIF.npFile, npt.Formats[npt.TF][npt.Fmaps],
                                     npFileSimF, simFormatIF, False))
   
    # WEIGHT
    tpbFilterShape = list(npt.reorderShape(npInfoW.npShape, npt.TF, npt.SIM, npt.Weights))
    (npFileSimW, simFormatW) = npt.copyNpyFileAs(npInfoW.npFile, npt.TF, npt.SIM, npt.Weights)
    tensorFormatMap.add(npInfoW.tensorName,
                        TensorFormat(npInfoW.tensorName, self.getOpName(),
                                    npInfoW.npFile, npt.Formats[npt.TF][npt.Weights],
                                    npFileSimW, simFormatW, True))

    fileList += [npFileSimW, npFileSim]
    stride = npt.reorderShape(self.getStrides(), npt.TF, npt.SIM, npt.Fmaps)
    padding = self.calcTpbPadding(self.getFilter2D(), self.getPaddingMode())
   

    layerData = {
      "layer_type"      : "Conv",
      "kernel_file"     : npFileSimW,
      "kernel_format"   : simFormatW,
      "kernel_shape"    : tpbFilterShape,
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormatOF,
      "ref_file"        : npFileSim,
      "padding"         : padding,
      "previous_layers" : [fromIfNode.getName()],
      "stride"          : stride,
      "#comment"        : "Two dimensional convolution with explicit padding"
    }
        
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def getWeightBytes(self):
    weightSizeBytes = 0
    if len(self.getNpInfo()) > 0:
      ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW)) = self.getInputNodesAndNpInfo()
      weightSizeBytes = npInfoW.nbytes()
    return weightSizeBytes

  # Node text for dot graph
  def getDotText(self):
    dotText = self.getOpType()
    if Config.Graph.showOpNameInKgraph:
      dotText += "\n" + self.getName()
    if len(self.getNpInfo()) > 0:
      dotText += "\nStrides " + str(self.getStrides())
      ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW)) = self.getInputNodesAndNpInfo()
      fmapSizeBytes = npInfoIF.nbytes()
      weightSizeBytes = npInfoW.nbytes()
      opCount = self.getOpCount()
      opsPerWeightByte = math.ceil(opCount / weightSizeBytes)
      # Data sizes
      npInfoOF = self.getNpInfo()[0]
      dotText += "\nw%.3f i%.3f o%.3f MiB" % (weightSizeBytes / 2**20,
                                           fmapSizeBytes / 2**20, npInfoOF.nbytes() / 2**20)
      dotText += "\nOpWB " + str(opsPerWeightByte)
      # Roofile, wavesize batch targets
      targetOpB = Config.Tpb.specTops*2**40 /(Config.Ddr.GiBps*2**30)/2 * Config.Tpb.numTpb
      targetBatchRoofLine = math.ceil(targetOpB / opsPerWeightByte)
      imgPixels = np.empty(self.getImg2D()).size
      targetBatchImgMin = math.ceil(Config.Pe.minWave / imgPixels)
      targetBatchImgOpt = math.floor(Config.Pe.maxWave / imgPixels)
      dotText += " BT(%d) %d-%d-%d" % (Config.Tpb.numTpb, targetBatchRoofLine,
                                       targetBatchImgMin, targetBatchImgOpt)
      # Ops
      dotText += "\nGop %.6f" % (opCount / 1e9)
      dotText += "\nGopPad %.6f" % (self.getOpCount(padded=True) / 1e9)
    else:
      dotText += "\n" + str(self.protoShape)
    return dotText

  # Number of add, multiply ops for performance analysis and reporting
  # E.g., 1 multiply and 1 accumulate is reported as 2 ops.
  def getOpCount(self, padded=False):
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (batch, channels, height, width) = tpbShape
    ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW)) = self.getInputNodesAndNpInfo()
    filterShapeRSCM = npInfoW.npShape
    # 7 loops based on the output tensor height * width - 4 in the filter, 3 in ofmap
    opCount = 2 * np.empty(filterShapeRSCM).size * batch * height * width;
    if not padded:
      # Adjust by area ratio of the padded and unpadded input
      padding = self.calcTpbPadding(self.getFilter2D(), self.getPaddingMode())
      u1,u2, (padNorth,padSouth), (padWest,padEast) = padding
      tpbShapeIn = list(npt.reorderShape(npInfoIF.npShape, npt.TF, npt.SIM, npt.Fmaps))
      (batchIn, channelsIn, heightIn, widthIn) = tpbShapeIn
      areaIn = heightIn * widthIn
      areaInPadded = (heightIn + padNorth + padSouth) * (widthIn + padWest + padEast)
      opCount *= 1.0 * areaIn / areaInPadded
    return opCount

  def isSupported(self):
    return True

###############################################################################
# 2D transposed convolution (aka deconvolution)
###############################################################################
class NodeConv2DTranspose(NodeBasePaddedStrided):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Return the 2 significant dimensions of batched multi channel IFMAP (of 2-D images)
  def getIFmapImageSize(self):
    (_, npInfoIF) = self.getInputNodesAndNpInfo()[2]
    ifmapArr = self.dim2imgSize(npInfoIF.npShape)
    return ifmapArr

  # Return the height and width dimensions of 2-D feature map
  def getImg2D(self):
    npInfo = self.getNpInfo()[0]
    img2D = npt.subShape(npInfo.npShape, "HW", npt.TF, npt.Fmaps)
    return img2D

  # Return the height and width dimensions of 2-D filter
  def getFilter2D(self):
    (_, npInfoW) = self.getInputNodesAndNpInfo()[1]
    filterArr = npt.subShape(npInfoW.npShape, "RS", npt.TF, npt.WeightsTrans)
    return filterArr

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (_, (_, npInfoW), (fromIfNode, npInfoIF)) = self.getInputNodesAndNpInfo()
    tpbFilterShape = list(npt.reorderShape(npInfoW.npShape, npt.TF, npt.SIM, npt.WeightsTrans))
    # OFMAP
    (npFileSim, simFormatOF) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    tensorFormatMap.add(npInfo.tensorName,
                        TensorFormat(npInfo.tensorName, self.getOpName(),
                                     npInfo.npFile, npt.Formats[npt.TF][npt.Fmaps],
                                     npFileSim, simFormatOF, False))
    # WEIGHT
    (npFileSimW, simFormatW) = npt.copyNpyFileAs(npInfoW.npFile, npt.TF, npt.SIM, npt.WeightsTrans)
    tensorFormatMap.add(npInfoW.tensorName,
                        TensorFormat(npInfoW.tensorName, self.getOpName(),
                                     npInfoW.npFile, npt.Formats[npt.TF][npt.WeightsTrans],
                                     npFileSimW, simFormatW, True))

    fileList += [npFileSimW, npFileSim]
    stride = npt.reorderShape(self.getStrides(), npt.TF, npt.SIM, npt.Fmaps)
    padding = self.calcTpbPadding(self.getFilter2D(), self.getPaddingMode())
    layerData = {
      "layer_type"      : "ConvTranspose",
      "kernel_file"     : npFileSimW,
      "kernel_format"   : simFormatW,
      "kernel_shape"    : tpbFilterShape,
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormatOF,
      "ref_file"        : npFileSim,
      "padding"         : padding,
      "previous_layers" : [fromIfNode.getName()],
      "stride"          : stride,
      "#comment"        : "Two dimensional transpose convolution (deconvolution) with explicit padding"
    }
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def getWeightBytes(self):
    weightSizeBytes = 0
    if len(self.getNpInfo()) > 0:
      (_, npInfoW) = self.getInputNodesAndNpInfo()[1]
      weightSizeBytes = npInfoW.nbytes()
    return weightSizeBytes

  # Node text for dot graph
  def getDotText(self):
    dotText = self.getOpType()
    if Config.Graph.showOpNameInKgraph:
      dotText += "\n" + self.getName()
    if len(self.getNpInfo()) > 0:
      dotText += "\nStrides " + str(self.getStrides())
      (_, (fromWeightNode, npInfoW), (fromIfNode, npInfoIF)) = self.getInputNodesAndNpInfo()
      fmapSizeBytes = npInfoIF.nbytes()
      weightSizeBytes = npInfoW.nbytes()
      opCount = self.getOpCount()
      opsPerWeightByte = math.ceil(opCount / weightSizeBytes)
      # Data sizes
      npInfoOF = self.getNpInfo()[0]
      dotText += "\nw%.3f i%.3f o%.3f MiB" % (weightSizeBytes / 2**20,
                                           fmapSizeBytes / 2**20, npInfoOF.nbytes() / 2**20)
      dotText += "\nOpWB " + str(opsPerWeightByte)
      # Roofile, wavesize batch targets
      targetOpB = Config.Tpb.specTops*2**40 /(Config.Ddr.GiBps*2**30)/2 * Config.Tpb.numTpb
      targetBatchRoofLine = math.ceil(targetOpB / opsPerWeightByte)
      imgPixels = np.empty(self.getImg2D()).size
      targetBatchImgMin = math.ceil(Config.Pe.minWave / imgPixels)
      targetBatchImgOpt = math.floor(Config.Pe.maxWave / imgPixels)
      dotText += " BT(%d) %d-%d-%d" % (Config.Tpb.numTpb, targetBatchRoofLine,
                                       targetBatchImgMin, targetBatchImgOpt)
      # Ops
      dotText += "\nGop %.6f" % (opCount / 1e9)
      dotText += "\nGopPad %.6f" % (self.getOpCount(padded=True) / 1e9)
    else:
      dotText += "\n" + str(self.protoShape)
    return dotText

  # Number of add, multiply ops for performance analysis and reporting
  # E.g., 1 multiply and 1 accumulate is reported as 2 ops.
  def getOpCount(self, padded=False):
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (batch, channels, height, width) = tpbShape
    (_, (fromWeightNode, npInfoW), (fromIfNode, npInfoIF)) = self.getInputNodesAndNpInfo()
    filterShapeRSCM = npInfoW.npShape
    # 7 loops based on the output tensor height * width - 4 in the filter, 3 in ofmap
    opCount = 2 * np.empty(filterShapeRSCM).size * batch * height * width;
    if not padded:
      # Adjust by area ratio of the padded and unpadded input
      padding = self.calcTpbPadding(self.getFilter2D(), self.getPaddingMode())
      u1,u2, (padNorth,padSouth), (padWest,padEast) = padding
      tpbShapeIn = list(npt.reorderShape(npInfoIF.npShape, npt.TF, npt.SIM, npt.Fmaps))
      (batchIn, channelsIn, heightIn, widthIn) = tpbShapeIn
      areaIn = heightIn * widthIn
      areaInPadded = (heightIn + padNorth + padSouth) * (widthIn + padWest + padEast)
      opCount *= 1.0 * areaIn / areaInPadded
    return opCount

  def isSupported(self):
    return True

###############################################################################
# Max, Avg Pool
###############################################################################
class NodePool(NodeBasePaddedStrided):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  def getKernelSize(self):
    return self.getAttr("ksize")

  # Return the 2 significant dimensions of Kernel Size
  def getKernelSize2D(self):
    kernelSizeNHWC = self.getKernelSize()
    kernelSize2D = kernelSizeNHWC[1:3]
    # Ensure 2-D filter
    assert len(kernelSizeNHWC) == 4
    assert kernelSize2D[0] > 0
    assert kernelSize2D[1] > 0
    return kernelSize2D

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (batch, channels, height, width) = tpbShape
    (fromIfNode, npInfoIF) = self.getInputNodesAndNpInfo()[0]
    kernelSizeNHWC = self.getKernelSize()
    kernelSizeNCHW = [kernelSizeNHWC[i] for i in [0, 3, 1, 2]]
    # OFMAP
    (npFileSim, simFormatOF) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    tensorFormatMap.add(npInfo.tensorName,
                        TensorFormat(npInfo.tensorName, self.getOpName(),
                                     npInfo.npFile, npt.Formats[npt.TF][npt.Fmaps],
                                     npFileSim, simFormatOF, False))
    # IFMAP, not needed
    (npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
    tensorFormatMap.add(npInfoIF.tensorName,
                        TensorFormat(npInfoIF.tensorName, self.getOpName(),
                                     npInfoIF.npFile, npt.Formats[npt.TF][npt.Fmaps],
                                     npFileSimF, simFormatIF, False))

    fileList += [npFileSim]
    stride = npt.reorderShape(self.getStrides(), npt.TF, npt.SIM, npt.Fmaps)
    padding = self.calcTpbPadding(self.getKernelSize2D(), self.getPaddingMode())
    layerData = {
      "layer_type"      : self.getOpType(),
      #"kernel_format"   : simFormatOF,  # redundant, has to be same as fmaps
      "kernel_shape"    : kernelSizeNCHW,
      "ofmap_shape"     : [batch, channels, height, width],
      "ofmap_format"    : simFormatOF,
      "ref_file"        : npFileSim,
      "padding"         : padding,
      "previous_layers" : [fromIfNode.getName()],
      "stride"          : stride,
      "#comment"        : "Two dimensional pool with explicit padding"
    }
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  # Node text for dot graph
  def getDotText(self):
    dotText = self.getOpType()
    if Config.Graph.showOpNameInKgraph:
      dotText += "\n" + self.getName()
    if len(self.getNpInfo()) > 0:
      # Data sizes
      npInfoOF = self.getNpInfo()[0]
      (fromIfNode, npInfoIF) = self.getInputNodesAndNpInfo()[0]
      dotText += "\ni%.3f o%.3f MiB" % (npInfoIF.nbytes() / 2**20,
                                       npInfoOF.nbytes() / 2**20)
      # Non-fusing cost: 2x the input
      # The cost is moving to the storage or back (DDR or SB)
      nfCostDdrUsec =   2 * npInfoIF.nbytes() / (Config.Ddr.GiBps*2**30) * 1e6
      dotText += "\nNonFuseDdr %.1f usec" %  nfCostDdrUsec
      nfCostSbUsec =   2 * npInfoIF.nbytes() / (Config.Tpb.poolGiBps*2**30) * 1e6
      dotText += "\nNonFuseSB %.1f usec" %  nfCostSbUsec

      # Kernel
      kernelSizeNHWC = self.getKernelSize()
      dotText += "\nKernelSize " + str(kernelSizeNHWC)
      # Stride
      dotText += "\nStrides " + str(self.getStrides())
    else:
      dotText += "\n" + str(self.protoShape)
    return dotText

  # Number of add, multiply, max or move, copy ops for performance
  # analysis and reporting
  # E.g., 1 multiply and 1 accumulate is reported as 2 ops.
  def getOpCount(self, padded=False):
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (batch, channels, height, width) = tpbShape
    kernelSizeNHWC = self.getKernelSize()
    opCount = 2 * np.empty(kernelSizeNHWC).size * batch * channels * height * width;
    if not padded:
      # Adjust by area ratio of the padded and unpadded input
      padding = self.calcTpbPadding(self.getKernelSize2D(), self.getPaddingMode())
      u1,u2, (padNorth,padSouth), (padWest,padEast) = padding
      (fromIfNode, npInfoIF) = self.getInputNodesAndNpInfo()[0]
      tpbShapeIn = list(npt.reorderShape(npInfoIF.npShape, npt.TF, npt.SIM, npt.Fmaps))
      (batchIn, channelsIn, heightIn, widthIn) = tpbShapeIn
      #strideVec = npt.reorderShape(self.getStrides(), npt.TF, npt.SIM, npt.Fmaps)
      #(batchStride, channelsStride, strideH, strideW) = strideVec
      areaIn = heightIn * widthIn
      areaInPadded = (heightIn + padNorth + padSouth) * (widthIn + padWest + padEast)
      opCount *= 1.0 * areaIn / areaInPadded
    return opCount


###############################################################################
# Basic 2-argument operations (e.g., Mul, Add) with no config (e.g., filter,
# stride, padding)
###############################################################################
class NodeSimple2(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []

    # Output tensor
    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)

    # Residual Add has both inputs dependent on the input image
    # BiasAdd has the other input constant
    # In Keras plain Add can be of either of the above types
    (faninEdge0, faninEdge1) = self.getFaninEdges()
    op0IsInMainDataFlow = faninEdge0.isInMainFlow()
    op1IsInMainDataFlow = faninEdge1.isInMainFlow()

    # If no main data flow, this op is a part of a side branch computation, so no layer needed
    # (TBD: do we need to enable scalar operations?)
    if not op0IsInMainDataFlow and not op1IsInMainDataFlow:
      return [], []

    # Join type if both incoming edges are main data flow edges
    isJoin = op0IsInMainDataFlow and op1IsInMainDataFlow 

    # Get predecessor nodes, keeping TF order (for Sub order is important)
    ((fromIfNode0, npInfoIF0), (fromIfNode1, npInfoIF1),) = self.getInputNodesAndNpInfo()
    (fromIfNodeMain, fromIfNodeSide) = (fromIfNode0, fromIfNode1)
    (npInfoIFMain, npInfoIFSide)     = (npInfoIF0, npInfoIF1)
    if op1IsInMainDataFlow:
      (fromIfNodeMain, fromIfNodeSide) = (fromIfNode1, fromIfNode0)
      (npInfoIFMain, npInfoIFSide)     = (npInfoIF1, npInfoIF0)

    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [fromIfNode0.getName(), fromIfNode1.getName()],
      "#comment"        : "Element-wise operation on two input tensors"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList

    # Override layer name to backend
    #   ResAdd - when both inputs depend on the input image
    overrideType = self.getOpType()
    if isJoin and overrideType == "Add":
      overrideType = "ResAdd"   # Maybe better to keep "Add"
    layerDataBase[0]["layer_type"] = overrideType

    if not isJoin:
      # Scalar is passed as a field (e.g. for LSTMs)
      if len(npInfoIFSide.npShape) == 0:
        val = npInfoIFSide.getValues()
        assert val.size == 1
        layerDataBase[0]["previous_layers"] = [fromIfNodeMain.getName()]
        if self.getOpType() == "ExpandDims":
          layerDataBase[0]["axis"] = np.asscalar(val.ravel()[0])
          layerDataBase[0]["#comment"] = "Insert new dimension of size 1 in the axis specified by \"axis\" field."
        else:
          layerDataBase[0]["add_scalar"] = np.asscalar(val.ravel()[0])
          layerDataBase[0]["#comment"] = "Element-wise operation on one input tensor and scalar."
      else:
        # Collapse the side node to a branch (except when it already is a real constant)
        # Main input is covered by a previous layer
        #   tfShape4D0 = npt.cShapeToNHWC(npInfoIF0.npShape)
        #   (npFileSimF0, simFormatIF0)  = npt.copyNpyFileAs(npInfoIF0.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D0)
        # Side input has to be collapsed to a constant
    
        (tpbShape1, simFormat1, npFileSim1) = self.convertShape(npInfoIFSide, tensorFormatMap, isConst=True)

        constLayerData = {
          "layer_type" :  "Const",
          "layer_name" :  fromIfNodeSide.getName(),
          "out_data_type"   : npInfo.dType,
          "ofmap_shape"     : tpbShape1,
          "ofmap_format"    : simFormat1,
          "ref_file"        : npFileSim1,
          "previous_layers" : [],
          "#comment"   :  "Captured constant"
        }
        fileListBase.insert(0, npFileSim1)
        layerDataBase.insert(0, constLayerData)  # prepend - because the backend Json parser requires layers defined

    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

###############################################################################
# MatMul - specialization for 2D data formats
###############################################################################
class NodeMatMul(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []

    # Output tensor is NC format
    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)
    
      # The IFMAP comes from reshape,  the other is (weight) matrix
    ((fromIfNode0, npInfoIF0), (fromIfNode1, npInfoIF1),) = self.getInputNodesAndNpInfo()
    
    prevNames = [fromIfNode0.getName()]
    (faninEdge0, faninEdge1) = self.getFaninEdges()
    isWeightsConst = not faninEdge1.isInMainFlow()    

    if isWeightsConst:
      # The matrix side input is handled like convolution weights
      (tpbShape4Dw, simFormatW, npFileSimW) = self.convertShapeForConstWeights(npInfoIF1, tensorFormatMap)
    else:
      # The matrix side input is handled like convolution weights
      (tpbShape4Dw, simFormatW, npFileSimW) = self.convertShape(npInfoIF1, tensorFormatMap)

      prevNames += [fromIfNode1.getName()]

    layerData = {
        # no kernel_file is needed.
        "kernel_format"   : simFormatW,
        "kernel_shape"    : tpbShape4Dw,
        "ofmap_shape"     : tpbShape,
        "ofmap_format"    : simFormat,
        "ref_file"        : npFileSim,
        "previous_layers" : prevNames,
        "#comment"        : "Two dimensional matrix multiply with two dynamic operands"
    }  
    if isWeightsConst:
      layerData["kernel_file"] = npFileSimW

    fileList.append(npFileSim)
    fileList.append(npFileSimW)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

  def getOpCount(self, padded=False):
    npInfo = self.getNpInfo()[0]
    tpbShape = self.getTpbShape(npInfo)
    (batch, channels, height, width) = tpbShape

    # The IFMAP comes from reshape,  the other is (weight) matrix
    ((fromIfNode0, npInfoIF0), (fromIfNode1, npInfoIF1),) = self.getInputNodesAndNpInfo()

    # Get the depth dimension of MatMul by treating the 2nd input as Fmap (unlike TPB's implementation)
    tpbShape1 = self.getTpbShape(npInfoIF1)
    (batch1_unused, channels1_unused, height1, width1_unused) = tpbShape1

    # 5 loops - batch, channels, and 3 for GEMM
    opCount = 2 * batch * channels * height * width * height1;
    return opCount

###############################################################################
# Mul - element-wise multiplication
# Unlike Matmul it uses Fmap format for both inputs
#
###############################################################################
class NodeMultiply(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []

    # LSTM Output tensor is NC format
    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)

    fileList.append(npFileSim)
    isScalar = [None]*2
    npInfoIF = [None]*2
    fromIfNode = [None]*2

    ((fromIfNode[0], npInfoIF[0]), (fromIfNode[1], npInfoIF[1]),) = self.getInputNodesAndNpInfo()
    # scalar (Mult, Min, Max) - one arg is scalar; element-wise: both are vectors
    isScalar[0] = len(npInfoIF[0].npShape) == 0
    isScalar[1] = len(npInfoIF[1].npShape) == 0
    assert(not (isScalar[0] and isScalar[1]))

    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [],
      "#comment"        : "Element-wise multiply two input tensors"
    }

    for i in (1, 0):
        if isScalar[i]:
            val = npInfoIF[i].getValues()
            assert val.size == 1
            layerData['mul_scalar'] = np.asscalar(val.ravel()[0])
        else:
            layerData["previous_layers"].insert(0, fromIfNode[i].getName()),

    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

###############################################################################
# Reshape
# Initial implementation is simply identity since data format conversions are done
# on all nodes
###############################################################################
class NodeReshape(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []

    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)

    # The IFMAP comes from reshape,  the other is (weight) matrix
    ((fromIfNode0, npInfoIF0), (fromIfNode1, npInfoIF1),) = self.getInputNodesAndNpInfo()

    # Both nodes are part of the main data flow
    # So use the tensor size to detect which one is reshape vector and which one IFMAP
    # Unsafe on 1x2 or 1x1 IFMAP
    assert any(np.empty(x.npShape).size > 2 for x in [npInfoIF0, npInfoIF1])
    if np.empty(npInfoIF0.npShape).size >  np.empty(npInfoIF1.npShape).size:
      ifNode = fromIfNode0
    else:
      ifNode = fromIfNode1

    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [ifNode.getName()],
      "#comment"        : "Reshape implemented as a copy operation (output dims limited to between 2-4)"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

###############################################################################
# Transpose
# Initial implementation is simply identity since data format conversions are done
# on all nodes
###############################################################################
class NodeTranspose(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []

    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)

    ((nIn, _), perm) = self.getInputNodesAndNpInfo()
    npInfoIndexinBes = 1
    permvec_tmp = perm[npInfoIndexinBes].getValues()
    permvec = [np.asscalar(permvec_tmp[i]) for i in range(len(permvec_tmp))]
    ordervec = [i for i in range(len(permvec_tmp))]
    permvec = list(map(lambda x,y: x - y, permvec, ordervec))
    if len(npInfo.npShape) == 4:
      permvec = npt.reorderShape(permvec, npt.TF, npt.SIM, npt.Fmaps)
    permvec = list(map(lambda x,y: x + y, permvec, ordervec))

    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "perm"            : permvec,
      "previous_layers" : [nIn.getName()],
      "#comment"        : "Transpose implemented as a copy operation (output dims limited to between 2-4)"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True


###############################################################################
# StridedSlice
#
###############################################################################
class NodeStridedSlice(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Node text for dot graph
  def getDotText(self):
    dotText = self.getOpType()
    if Config.Graph.showOpNameInKgraph:
      dotText += "\n" + self.getName()
    for attrName in ["begin_mask", "ellipsis_mask", "end_mask", "new_axis_mask", "shrink_axis_mask"]:
      attrVal = self.getAttr(attrName)
      if not attrVal == None:
        dotText += "\n%s : %g" % (attrName, attrVal)
    if len(self.getNpInfo()) > 0:
      bes = {"Begin" : (), "End" : (), "Stride" : ()}
      ((nIn, npiIn), bes["Begin"], bes["End"], bes["Stride"]) = self.getInputNodesAndNpInfo()
      dotText += "\nShapeIn : %s" % str(npiIn.npShape)
      for c in ["Begin", "End", "Stride"]:
        (fromNode, fromNpInfo) = bes[c]
        npVal = fromNpInfo.getValues()
        dotText += "\n%s : %s" % (c, str([int(i) for i in npVal]))
    else:
      dotText += "\n" + str(self.protoShape)
    return dotText

  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    bes = {"Begin" : (), "End" : (), "Stride" : ()}
    ((nIn, npInfoFrom), bes["Begin"], bes["End"], bes["Stride"]) = self.getInputNodesAndNpInfo()
    npInfoIndexinBes = 1

    # Suppress StridedSlice in constant or reshape calculations in CNNs
    # FIX_THIS: this should be a graph transform
    if len(npInfo.npShape) == 1:
      return {},[]

    # get constants
    # shift to get to
    n_dims = len(npInfo.npShape)
    # This is for splitting along channel Axis - -- for --  Resnet???
    begin_indices = bes["Begin"][npInfoIndexinBes].getValues().tolist()
    end_indices = bes["End"][npInfoIndexinBes].getValues().tolist()
    # shift uint8 to get at masks shifted to the right and then reverse
    # for indexing
    begin_mask = np.unpackbits(np.array([self.getAttr("begin_mask")], \
        dtype=np.uint8)).tolist()[8-n_dims:][::-1]
    end_mask = np.unpackbits(np.array([self.getAttr("end_mask")], \
        dtype=np.uint8)).tolist()[8-n_dims:][::-1]
    to_nhwc = lambda l, val : None
    if n_dims == 4:
      tfFormat = npt.NHWC
      tfShape4D = npInfo.npShape
      to_nhwc = lambda l, val : None
    elif n_dims == 3: #NWC
      tfFormat = npt.NWC
      tfShape4D = npt.nwcShapeToNHWC(npInfo.npShape)
      to_nhwc = lambda l, val : l.insert(2, val)
    elif n_dims == 2: #NC
      if not Config.Graph.useWCFormat:
        tfFormat = npt.NC
        tfShape4D = npt.ncShapeToNHWC(npInfo.npShape)
        to_nhwc = lambda l, val : l.insert(1, val) or l.insert(1, val)
      else:
        tfFormat = npt.WC
        tfShape4D = npt.wcShapeToNHWC(npInfo.npShape)
        to_nhwc = lambda l, val : l.insert(0, val) or l.insert(0, val)       
    else: #C
      tfFormat = npt.C
      tfShape4D = npt.cShapeToNHWC(npInfo.npShape)
      to_nhwc = lambda l, val : l.insert(0, val) or l.insert(0, val) or l.insert(0, val)
    # take not of channel axi in original input
    channel_axis = tfFormat.find('C')
    (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D)

    # get everything to nchw format by going through nhwc
    to_nhwc(begin_indices, 0)
    to_nhwc(end_indices, 0)
    to_nhwc(begin_mask, 1)
    to_nhwc(end_mask, 1)

    to_nchw = lambda l: list(npt.reorderShape(l, npt.TF, npt.SIM, npt.Fmaps))
    begin_indices = to_nchw(begin_indices)
    end_indices = to_nchw(end_indices)
    begin_mask = to_nchw(begin_mask)
    end_mask = to_nchw(end_mask)
    tfShape4D = to_nchw(tfShape4D)

    tensorFormatMap.add(npInfo.tensorName,
                        TensorFormat(npInfo.tensorName, self.getOpName(),
                                     npInfo.npFile, tfFormat,
                                     npFileSim, simFormat, False))
    # we are in NCHW now
    vectorStart = begin_indices[1]
    vectorEnd = end_indices[1]
    if vectorEnd <= vectorStart:
      vectorEnd = npInfoFrom.npShape[channel_axis]
      assert self.getAttr("end_mask") > 0
    assert all(bes['Stride'][npInfoIndexinBes].getValues()) == 1
    layerData = {
      "ofmap_shape"     : tfShape4D,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "channel_slice"   : [vectorStart, vectorEnd],
      "begin_mask"      : begin_mask,
      "begin_indices"   : begin_indices,
      "end_mask"        : end_mask,
      "end_indices"     : end_indices,
      "previous_layers" : [nIn.getName()],
      "#comment"        : "Extract slice along channel dimension"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

###############################################################################
# Slice
#
###############################################################################
class NodeSlice(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Node text for dot graph
  def getDotText(self):
    dotText = self.getOpType()
    if Config.Graph.showOpNameInKgraph:
      dotText += "\n" + self.getName()
    if len(self.getNpInfo()) > 0:
      bes = {"Begin" : (), "Size" : ()}
      ((nIn, npiIn), bes["Begin"], bes["Size"]) = self.getInputNodesAndNpInfo()
      dotText += "\nShapeIn : %s" % str(npiIn.npShape)
      for c in ["Begin", "Size"]:
        (fromNode, fromNpInfo) = bes[c]
        npVal = fromNpInfo.getValues()
        dotText += "\n%s : %s" % (c, str([int(i) for i in npVal]))
    else:
      dotText += "\n" + str(self.protoShape)
    return dotText

  def genCompilerLayerJson(self, tensorFormatMap):
    axis = self.getAttr("axis")
    fileList = []
    npInfo = self.getNpInfo()[0]
    bes = {"Begin" : (), "Size" : ()}
    inNodesAndNpInfos = self.getInputNodesAndNpInfo()
    if len(inNodesAndNpInfos) == 3:
        ((nIn, _), bes["Begin"], bes["Size"]) = inNodesAndNpInfos
        npInfoIndexinBes = 1
        sliceBegin_tmp = bes["Begin"][npInfoIndexinBes].getValues()
        sliceBegin = [np.asscalar(sliceBegin_tmp[i]) for i in range(sliceBegin_tmp.shape[0])]
        sliceSize_tmp = bes["Size"][npInfoIndexinBes].getValues()
        sliceSize = [np.asscalar(sliceSize_tmp[i]) for i in range(sliceSize_tmp.shape[0])]
    elif len(inNodesAndNpInfos) == 1:  ## This Slice is from Unstack
        (nIn, _) = inNodesAndNpInfos[0]
        sliceBegin = self.getAttr("slice_begin")
        sliceSize = self.getAttr("slice_size")
    else:
        assert(False)

    # Suppress StridedSlice in constant or reshape calculations in CNNs
    # FIX_THIS: this should be a graph transform
    if len(npInfo.npShape) == 1:
      return {},[]

    sliceBegin = self.convertDimParams(sliceBegin, 0)
    sliceSize = self.convertDimParams(sliceSize, -1)
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)

    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "slice_begin"     : sliceBegin,
      "slice_size"      : sliceSize,
      "previous_layers" : [nIn.getName()],
      "#comment"        : "Extract slice along channel dimension"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True


###############################################################################
# Split
#
###############################################################################
class NodeSplit(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    if len(npInfo.npShape) == 4:
      tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
      (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
      tfFormat = npt.Formats[npt.TF][npt.Fmaps]
    else:
      tfShape4D = npt.ncShapeToNHWC(npInfo.npShape)
      (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D)
      tpbShape = list(npt.reorderShape(tfShape4D, npt.TF, npt.SIM, npt.Fmaps))
      tfFormat = npt.NC

    # extract input values
    # FIXME: this works for amoebanetd, not clear if it is generalized
    INAndNI = self.getInputNodesAndNpInfo()
    assert len(INAndNI) == 2, "More inputs than I expected!"

    splitDimVal = np.asscalar(INAndNI[0][1].getValues()[0])

    numSplitVal = self.getAttr("num_split")

    # get real input node
    (fromIfNode, npInfoIF) = INAndNI[1]
    tensorFormatMap.add(npInfo.tensorName,
                        TensorFormat(npInfo.tensorName, self.getOpName(),
                                     npInfo.npFile, tfFormat,
                                     npFileSim, simFormat, False))

    nextLayerPosList = self.getFanoutNodePosNames()

    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "num_splits"      : numSplitVal,
      "axis"            : splitDimVal,
      "previous_layers" : [fromIfNode.getName()],
      "next_layer_order" : nextLayerPosList,
      "#comment"        : "Split a tensor into subtensors.  see https://www.tensorflow.org/api_docs/python/tf/split"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True


###############################################################################
# Unstack
#
###############################################################################
class NodeUnstack(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    if len(npInfo.npShape) == 4:
      tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
      (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
      tfFormat = npt.Formats[npt.TF][npt.Fmaps]
    else:
      tfShape4D = npt.ncShapeToNHWC(npInfo.npShape)
      (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D)
      tpbShape = list(npt.reorderShape(tfShape4D, npt.TF, npt.SIM, npt.Fmaps))
      tfFormat = npt.NC
    ((fromIfNode, npInfoIF),) = self.getInputNodesAndNpInfo()

    tensorFormatMap.add(npInfo.tensorName,
                        TensorFormat(npInfo.tensorName, self.getOpName(),
                                     npInfo.npFile, tfFormat,
                                     npFileSim, simFormat, False))

    unstackAxis = self.getAttr("axis")
    nextLayerPosList = self.getFanoutNodePosNames()

    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "unstack_axis"    : unstackAxis,
      "previous_layers" : [fromIfNode.getName()],
      "next_layer_order" : nextLayerPosList,
      "#comment"        : "Unstack along dimension specified by unstack_axis, using stacking order next_layer_order"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)


  def isSupported(self):
    return True

###############################################################################
# ClipByValue
#
###############################################################################
class NodeClipByValue(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    if len(npInfo.npShape) == 3:
      tfShape4D = npt.nwcShapeToNHWC(npInfo.npShape)
      tfFormat = npt.NWC
    elif len(npInfo.npShape) == 2:
      tfShape4D = npt.ncShapeToNHWC(npInfo.npShape)
      tfFormat = npt.NC
    elif len(npInfo.npShape) == 1:
      tfShape4D = npt.cShapeToNHWC(npInfo.npShape)
      tfFormat = npt.C
    else:
      assert len(npInfo.npShape) == 4
      tfShape4D = npInfo.npShape
      tfFormat = npt.Formats[npt.TF][npt.Fmaps]

    tpbShape = list(npt.reorderShape(tfShape4D, npt.TF, npt.SIM, npt.Fmaps))
    (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D)
    tensorFormatMap.add(npInfo.tensorName,
                        TensorFormat(npInfo.tensorName, self.getOpName(),
                                     npInfo.npFile, tfFormat,
                                     npFileSim, simFormat, False))

    bes = {"clip_value_min" : (), "clip_value_max" : ()}
    ((nIn, npInfoFrom), bes["clip_value_min"], bes["clip_value_max"]) = self.getInputNodesAndNpInfo()
    npInfoIndexinBes = 1
    clip_val_min = np.asscalar(bes["clip_value_min"][npInfoIndexinBes].getValues())
    clip_val_max = np.asscalar(bes["clip_value_max"][npInfoIndexinBes].getValues())

    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "clip_value_max"  : clip_val_max,
      "clip_value_min"  : clip_val_min,
      "previous_layers" : [nIn.getName()],
      "#comment"        : "Clip values at min and max values"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

###############################################################################
# Pad (explicit padding)
#
###############################################################################
class NodePad(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]

    ((nIn, npInfoFrom), padding_inp) = self.getInputNodesAndNpInfo()
    npInfoIndexinBes = 1
    padvec_tmp = padding_inp[npInfoIndexinBes].getValues()
    padvec = [[np.asscalar(padvec_tmp[i][j]) for j in range(padvec_tmp.shape[1])] for i in range(padvec_tmp.shape[0])]

    if len(npInfo.npShape) == 4:
      # CNN unit test flow, no known large Tonga NNs use these shapes as of early 2018
      tfShape4D = npInfo.npShape
      tfFormat = npt.Formats[npt.TF][npt.Fmaps]
      padvec = npt.reorderShape(padvec, npt.TF, npt.SIM, npt.Fmaps)
    elif len(npInfo.npShape) == 3:
      tfShape4D = npt.nwcShapeToNHWC(npInfo.npShape)
      tfFormat = npt.NWC
      padvec.insert(1, [0,0])
      padvec = npt.reorderShape(padvec, npt.TF, npt.SIM, npt.Fmaps)
    elif len(npInfo.npShape) == 2:
      tfShape4D = npt.ncShapeToNHWC(npInfo.npShape)
      tfFormat = npt.NC
      padvec.insert(1, [0,0])
      padvec.insert(1, [0,0])
      padvec = npt.reorderShape(padvec, npt.TF, npt.SIM, npt.Fmaps)
    else:
      raise RuntimeError("Supported number of dimensions for Pad's output tensor is between 2 and 4; found %d dimensions instead"%(len(npInfo.npShape)))

    tpbShape = list(npt.reorderShape(tfShape4D, npt.TF, npt.SIM, npt.Fmaps))
    (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D)
    tensorFormatMap.add(npInfo.tensorName,
                        TensorFormat(npInfo.tensorName, self.getOpName(),
                                     npInfo.npFile, tfFormat,
                                     npFileSim, simFormat, False))

    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "padding"         : padvec,
      "previous_layers" : [nIn.getName()],
      "#comment"        : "Pad FMAP using low/high values from 2 elem array per dimension"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True

###############################################################################
# A node for ReduceOp operation (supports only reduce_sum for now)
###############################################################################
class NodeReduceOp(Node):
  def __init__(self, name, opType, attrs):
    assert(opType == 'Sum')
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self, tensorFormatMap):
    fileList = []
    npInfo = self.getNpInfo()[0]
    (tpbShape, simFormat, npFileSim) = self.convertShape(npInfo, tensorFormatMap)
    ((fromIfNode, npInfoIF), (fromIdxNode, npInfoIdx)) = self.getInputNodesAndNpInfo()

    axesTF = [0, 0, 0, 0] # NHWC
    if len(npInfoIF.npShape) == 2:
      if Config.Graph.useWCFormat:
        for v in npInfoIdx.getValues():
          if v == -1 or v == 1:
            axesTF[3] = 1 # C axis
          elif v == 0:
            axesTF[2] = 1 # W axis
          else:
            assert(False and 'invalid reduce axis')
      else:
        # NC format
        for v in npInfoIdx.getValues():
          if v == -1 or v == 1:
            axesTF[3] = 1 # C axis
          elif v == 0:
            axesTF[0] = 1 # N axis
          else:
            assert(False and 'invalid reduce axis')
    elif len(npInfoIF.npShape) == 4:
      for v in npInfoIdx.getValues():
        axesTF[3 if v == -1 else v] = 1
    else:
      assert(False and 'unsupported dimension')

    axesNCHW = list(npt.reorderShape(axesTF, npt.TF, npt.SIM, npt.Fmaps))            
    
    layerData = {
      "reduce_axes"     : axesNCHW,
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [fromIfNode.getOpName()],
      "#comment"        : "Reduce tensor along with given axes"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self, tensorFormatMap)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True          

###############################################################################
# Computational data flow graph
###############################################################################
class Graph(Object):
  def __init__(self, name = "GRAPH", attrs = {}, schedulerMode = "tcc", debugLevel=0):
    super().__init__(name, attrs)
    self.__name2node = {}
    self.__edges = []
    self.__mainFlowEdges = []
    self.__inputNodes = []
    self.kaenaPath = os.environ["KAENA_PATH"]
    self.schedulerMode = schedulerMode
    self.debugLevel = debugLevel
    self.tensorFormatMap = TensorFormatMap()

  def setSchedulerMode(self, mode):
    self.schedulerMode = mode

  def addNode(self, node):
    self.__name2node[node.getName()] = node

  def addEdge(self, fromName, fromIndex, toName, toIndex, attrs = {}):
    fromNode = self.getNode(fromName)
    toNode = self.getNode(toName)
    lattrs = attrs
    #lattrs.update({"label" : "E" + str(len(self.__edges))})
    edge = Edge(PosNode(fromNode, fromIndex), PosNode(toNode, toIndex), lattrs)
    #print("DEBUG added edge with attrs ", edge.getAttrs())
    self.__edges.append(edge)
    fromNode.setFanoutEdge(edge, fromIndex)
    toNode.setFaninEdge(edge, toIndex)
    return edge

  def hasNode(self, name):
    return(name in self.__name2node)

  def getNode(self, name):
    return(self.__name2node[name])

  def getNodes(self):
    return(list(self.__name2node.values()))

  def getEdges(self):
    return(self.__edges)

  def deleteEdge(self, edge):
        fromPosNode = edge.getFromPosNode()  ## A
        toPosNode = edge.getToPosNode()   ## B
        fromPosNode.node.disconnectFanoutEdge(fromPosNode.index, edge)
        toPosNode.node.disconnectFaninEdge(edge)
        self.__edges.remove(edge)

  def nodeSuccessors(self, fromNode):
    nextNodes = {}
    for edge in fromNode.getFanoutEdges():
      nextNodes[edge.getToPosNode().node] = 1
    return(list(nextNodes.keys()))

  def nodePredecessors(self, toNode):
    preNodes = []
    for edge in toNode.getFaninEdges():
      if edge == None:
        print("INTERNAL WARNING: nodePredecessors of Node %s has None edge" % toNode.getName())
      else:
        preNodes.append(edge.getFromPosNode().node)
    return(preNodes)

  # Get the nodes with no successors - highest in the data flow level
  def getTopNodes(self):
    nextNodes = self.getNodes()
    topNodes = []
    visitedNodes = set()
    while len(nextNodes) > 0:
      newNextNodes = []
      for n in nextNodes:
        if not n in visitedNodes:
          nodeSuccessors = self.nodeSuccessors(n)
          if len(nodeSuccessors) > 0:
            newNextNodes += nodeSuccessors
          else:
            topNodes.append(n)
          visitedNodes.add(n)
      nextNodes = list(set(newNextNodes))
      assert len(topNodes) == len(list(set(topNodes)))
    return topNodes

  # Get the node with most computation - highest in the data flow level
  def getTopNode(self):
    #nextNodes = self.getNodes()
    #while len(nextNodes) > 0:
    #  n = nextNodes[0]
    #  nextNodes = self.nodeSuccessors(n)
    #return(n)
    return self.getTopNodes()[0]

  def setInputNodes(self, nodeList):
    self.__inputNodes = nodeList
  def getInputNodes(self):
    return(self.__inputNodes)
  # Legacy API
  def getInputNode(self):
    print("WARNING: using legacy API getInputNode")
    return(self.__inputNodes[0])

  # On a levelized graph - max depth to reach node among all paths
  # It describes "computational readiness" in the data flow:
  #   all ops at level N done implies that an op at level N+1 can
  #   be safely executed

  def levelize(self):
    postponedCounts = {}
    nodes = self.getNodes()
    postponedLimit = max(1000, len(nodes))
    perDot = max(1, int(len(nodes) / 80))
    print("INFO: levelizing ...")
    self.__level2nodes = []
    numLevelized = 0
    while len(nodes) > 0:
      n = nodes.pop(0)
      #print("DEBUG: levelize popped node  ", n.getName())
      level = 0
      for preNode in self.nodePredecessors(n):
        if preNode.getLevel() < 0:
          #print("    DEBUG: postponed node  ", n.getName())
          nodes.append(n)
          postponedCounts[n] = 1 + postponedCounts.get(n, 0)
          if postponedCounts[n] > postponedLimit:
            raise RuntimeError("\nERROR: levelize failed likely due to a loop involving node %s. Rerun without --images, and review the TF IR svg" % n)
          level = -1
          break
        else:
          level = max(level, preNode.getLevel())
      if level >= 0:
        level += 1
        n.setLevel(level)
        #print("  DEBUG: node ", n.getName(), " got level ", level)
        while len(self.__level2nodes) <= level:
          self.__level2nodes.append([])
        self.__level2nodes[level].append(n)
        numLevelized += 1
        if numLevelized % perDot == 0:
          print(".", end='', flush=True)
    print("", flush=True)

  # array of node lists, index is level from leaves (inputs) to output (classify)
  def getLevelizedNodes(self):
    return(self.__level2nodes)

  # Marks edges that are important for visualization and data flow transformations
  # Typically it is transitive fanout of the input tensor
  def identifyMainFlowEdges(self):
    nodes = {n : 0 for n in self.getInputNodes()}
    assert(len(nodes.keys()) > 0)
    nodeFront = nodes.copy()
    while len(nodeFront) > 0:
      nodeFrontNew = {}
      for n in nodeFront.keys():
        #if n.getName() == "resnet_v2_152/block1/unit_1/bottleneck_v2/add":
        #  print("DEBUG")
        for nextNode in self.nodeSuccessors(n):
          if nextNode not in nodes:
            nodes[nextNode] = 0
            nodeFrontNew[nextNode] = 0
      nodeFront = nodeFrontNew
    for n in nodes.keys():
      self.__mainFlowEdges += n.getFanoutEdges()
    for e in self.__mainFlowEdges:
      e.setIsInMainFlow(True)

  def edgeIsInMainFlow(self, edge):
    return(edge in self.__mainFlowEdges)

  def genCompilertgz(self, outTgzFile, fileList):
    # Tar all files as package
    cmd = ("tar cvzf %s " % outTgzFile) + " ".join(fileList)
    print(getInfoDateStr(), "executing  %s" %cmd)
    os.system(cmd)

  ##############################################################################
  # outIdx is the index in the __fanout (array of array of edges)
  def CreateOneSliceNodeFromUnstack(self, unstackNode, driverPosNode, outIdx,
                                    simInputUnstackAxisInt):
        outputs = unstackNode.getFanouts()
        unstackOutNpInfos = unstackNode.getNpInfo()
        numDims = len(unstackOutNpInfos[outIdx].npShape)

        outEdges = unstackNode.getFanoutEdgesOfOutput(outIdx)
        assert(len(outEdges) > 0)
        outPinIdx = outEdges[0].getFromPosNode().index
        for outEdge in outEdges:
            assert(outEdge.getFromPosNode().index == outPinIdx)
        outPinNpInfo = unstackOutNpInfos[outPinIdx]

        unstackName = unstackNode.getName()
        sliceName = unstackName + "_slc_" + str(outPinIdx)

        sizes = []
        begins = []
        for i in range(numDims):
            if i == simInputUnstackAxisInt:
                beg = outIdx
                siz = 1
            else:
                beg = 0
                siz = -1
            sizes.append(siz)
            begins.append(beg)
        sliceAttrs = {}
        sliceAttrs["slice_begin"] = begins
        sliceAttrs["slice_size"] = sizes
        sliceAttrs["in_sim_unstack_axis"] = simInputUnstackAxisInt

        sliceNode = NodeSlice(sliceName, "Slice", sliceAttrs)
        self.addNode(sliceNode)

        ##############################
        ## input side of unstack
        edgeAttrs = {}
        self.addEdge(driverPosNode.node.getName(), driverPosNode.index, sliceNode.getName(), 0, {})

        ##############################
        ## output side of unstack
        sliceNode.appendNpInfo(unstackOutNpInfos[outPinIdx])

        oldEdges = []

        toNodeNames = []
        toIndices = []
        for edge in outEdges:
            oldEdges.append(edge)
            toNodeNames.append(edge.getToPosNode().node.getName())
            toIndices.append(edge.getToPosNode().index)

        for edge in oldEdges:
            self.deleteEdge(edge)

        for i in range(len(toNodeNames)):
            toNodeName = toNodeNames[i]
            toIndex = toIndices[i]
            newEdge = self.addEdge(sliceNode.getName(), 0,
                                   toNodeName, toIndex, {})

        return sliceNode



  ################################################################################
  def findSimInputUnstackAxis(self, unstackNode):
        inEdge0 = unstackNode.getFaninEdges()[0]
        driverPin = inEdge0.getFromPosNode()
        driverNode = driverPin.node
        driverTensorName = driverPin.getName()

        ## Calling genCompilerLayerJson() is unfortunate. There should
        ## be a lighter method to get format of tensor
        driverNode.genCompilerLayerJson(self.tensorFormatMap)

        driverTensorFormat = self.tensorFormatMap.get(driverTensorName)
        assert(driverTensorFormat)
        driverTensorTfFormat = driverTensorFormat.tfFormat
        driverTensorSimFormat = driverTensorFormat.simFormat

        tfInputUnstackAxisInt = unstackNode.getAttr("axis")
        ## Find the letter for the axis (e.g., one of N, H, C, or W)
        inputUnstackAxisName  = driverTensorTfFormat[tfInputUnstackAxisInt]
        ## Find the axis for the letter, but in sim format for in tensor
        simInputUnstackAxisInt = -1
        for n in range(len(driverTensorSimFormat)):
            if driverTensorSimFormat[n] == inputUnstackAxisName:
                simInputUnstackAxisInt = n
                break
        return simInputUnstackAxisInt

  ################################################################################
  # Replace
  #     A --> B(Unstack) --> C
  #                      +-> D
  # with
  #     A --> B1(slice) --> C
  #       +-> B2(slice) --> D
  ################################################################################
  # Graph method
  def ReplaceOneUnstackNode(self, levelIdx, unstackNode):
        simInputUnstackAxisInt = self.findSimInputUnstackAxis(unstackNode)
        assert(simInputUnstackAxisInt >= 0)


        levelizedNodes = self.getLevelizedNodes()
        level = levelizedNodes[levelIdx]
        ## Number of NpInfos can be larger than the number of sublists
        ## in the Fanout list
        outputs = unstackNode.getFanouts()
        numOutputs = len(outputs)
        unstackNodeNpInfos = unstackNode.getNpInfo()

        assert(numOutputs <= len(unstackNodeNpInfos))

        ## All output tensors have the same shape in Unstack
        for np_info in unstackNodeNpInfos:
            assert(np_info.npShape == unstackNodeNpInfos[0].npShape)

        ## disconnect out edge from driver
        inEdges = unstackNode.getFaninEdges()
        assert(len(inEdges) == 1)
        inEdge = inEdges[0]

        driverPosNode = inEdge.getFromPosNode()  ## A
        self.deleteEdge(inEdge)

        sliceNodes=[]

        for outIdx in range(numOutputs):
            sliceNode = self.CreateOneSliceNodeFromUnstack(unstackNode,
                                            driverPosNode, outIdx,
                                            simInputUnstackAxisInt)
            sliceNodes.append(sliceNode)

        return sliceNodes



  ################################################################################
  # Graph method
  def ReplaceUnstackNodes(self):
        levelizedNodes = self.getLevelizedNodes()
        for levelIdx in range(0, len(levelizedNodes)):
            level = levelizedNodes[levelIdx]
            oldNodes = []
            newNodes = []
            for node in levelizedNodes[levelIdx]:
                if not node.isSupported() or not isinstance(node, NodeUnstack):
                    continue
                print("Replacing unstack node", node)
                oldNodes.append(node)
                slicedNodes = self.ReplaceOneUnstackNode(levelIdx, node)
                newNodes += slicedNodes

            for node in oldNodes:
                level.remove(node)
            level += newNodes


  ################################################################################
  # Generate Kaena Compiler input graph (in json format)
  # including weights, input and golden output imges (in numpy format)
  #
  # NPY file formats:
  #   source (in TF) -> conversions to Inkling/Tonga -> NPY written by TFFE
  #
  # IFMAPS:
  #   NHWC  -> NCHW
  #
  # WEIGHTS:
  #   RSCM   ->  MCRS
  #
  # Returns npy reference file for the last layer, and list of files to package into K-graph tgz

  def genCompilerJson(self, outFile, verbose):

    self.ReplaceUnstackNodes()

    jsonData = {
      "net_name"  : "TrivNet",
      "layers"   : []
    }
    fileList = []

    # Input layers
    inputNodes = self.getInputNodes()
    for inputNode in self.getInputNodes():
      npInfo = inputNode.getNpInfo()[0]
      jsonData["data_type"] = npInfo.dType   # No conversion by npu.dtypeToStr() was needed

      (tpbShape, simFormat, npFileSim) = inputNode.convertShape(npInfo, self.tensorFormatMap)

    outNpy = npFileSim
    # Conv and other layers
    levelizedNodes = self.getLevelizedNodes()
    totalOpCount = 0
    totalOpCountPadded = 0
    layers = OrderedDict()
    fileLists = OrderedDict()
    for level in range(0, len(levelizedNodes)):

      nodesAtLevelOrig = list(levelizedNodes[level])
      nodesAtLevel = nodesAtLevelOrig.copy()
      if Config.levelOrderSeed == None:
        nodesAtLevel = sorted(nodesAtLevelOrig)
      else:
        random.Random(Config.levelOrderSeed).shuffle(nodesAtLevelOrig)
      if nodesAtLevel != nodesAtLevelOrig:
        print("INFO: reordered nodes at level %d to %s" % (level, [n.getName() for n in nodesAtLevel]))
      for n in nodesAtLevel:
        nodeName = n.getName()
        #print("DEBUG graph::genCompilerJson: node=", n.getName(), "  op=", op)
        #isMainFlowConst = (n.getOpType() == "Const" and any(ns.isMainFlowNode() for ns in self.nodeSuccessors(n)))
        #print("  DEBUG const and successor in main flow " + str(isMainFlowConst))
        if not n.isSupported():
          print("WARNING: skipping layer data for layer: %s" % n.getName())
        else:
          (layerData, fileListLayer) = n.genCompilerLayerJson(self.tensorFormatMap)
          if Config.debugLevel >= 1:
            print("DEBUG: adding layer data for layer: %s, type %s" % (n.getName(), type(n)))

          # Resadd captured constant overrides input layer to work around the
          # current BE requirement that BiasAdd has the bias vector as a constant.
          # This is a hack
          for l in layerData:
            layers[l["layer_name"]] = l
            fileLists[l["layer_name"]] = fileListLayer
        opCount = n.getOpCount()
        opCountPadded = n.getOpCount(padded=True)
        totalOpCount += opCount
        totalOpCountPadded += opCountPadded
        if Config.debugLevel >= 1:
          print("DEBUG: opcount is %d for %s  %s" % (opCount, n.getOpType(), n.getOpName()))
          print("DEBUG: padded opcount is %d for %s  %s" % (opCountPadded, n.getOpType(), n.getOpName()))

    for l in layers:
      jsonData["layers"] += [layers[l]]
      fileListLayer = fileLists[l]
      fileList += fileListLayer
      outNpy = fileListLayer[-1]

    print("INFO: total opcount is %d" % totalOpCount)
    print("INFO: total padded opcount is %d" % totalOpCountPadded)


    if verbose > 0:
      npu.showNpyFile("Output OFMAPs", outNpy)

    with open(outFile, "w") as f:
      s = json.dumps(jsonData, indent=2, sort_keys=True)
      ## Blank space formatting - Remove redundant new lines on list
      ##   of integers, and list of pairs of integers.
      s = re.sub(r'\s+(\d+,)\n', r' \1', s, flags=re.S)
      s = re.sub(r',\s+(\d+)\n\s+\]', r', \1 ]', s, flags=re.S)
      s = re.sub(r'\s+(\[ \d+, \d+ \],)\n', r' \1', s, flags=re.S)
      s = re.sub(r',\s+(\[ \d+, \d+ \])\n\s+\]', r', \1 ]', s, flags=re.S)
      f.write(s)
    print("INFO: wrote ", outFile)
    fileList.append(outFile)
    self.tensorFormatMap.writeJson("tensor_map.json")

    return(outNpy, fileList)


  def getLowestLevelNodes(self):
    return self.getLevelizedNodes()[1]  # levels start from 1

  def print(self):
    for n in self.getNodes():
      print("  Node  %-12s  %s" % (n.getOpType(), n.getName()))
    for e in self.getEdges():
      print("  Edge  %s  ->  %s" % (e.getFromNode().getName(),
                                    e.getToNode().getName()))

  # Copy edge from another graph into this one
  def copyEdge(self, e):
    fromPosNode = e.getFromPosNode()
    toPosNode = e.getToPosNode()
    fromName = fromPosNode.node.getName()
    toName = toPosNode.node.getName()
    if self.hasNode(fromName) and self.hasNode(toName):
      eNew = self.addEdge(fromName, fromPosNode.index,
                   toName, toPosNode.index, e.getAttrs())
      eNew.setLabel(e.getLabel())
      eNew.setIsInMainFlow(e.isInMainFlow())
      if Config.debugLevel >= 1:
        print("DEBUG: copyEdge %s %d -> %s %d" %
              (fromName, fromPosNode.index, toName, toPosNode.index))
      return eNew
    return None

  # Copy edges from a graph into this one based on node existance in this one
  def copyEdges(self, sourceGraph):
    for e in sourceGraph.getEdges():
      self.copyEdge(e)

  # Writes graph in a given format using graphviz lib
  # Key operations like convolution are colored red
  # The depth determines subgraph clastering based on operation names
  #   layerN/branchM/convP cluster would be breated (in blue) if depth is 3
  def writeDot(self, depth, outFile, outFormat = "svg"):
    dot = Digraph(comment="writeDot")
    dot.node("KgraphLegend", "Legend" + re.sub("\n", "\l", Config.Graph.legendText),
             {"color":"yellow", "shape":"rectangle"})
    for n in self.getNodes():
      opType = n.getOpType()
      attrs = {}
      if re.search("conv", opType, re.I):
        attrs["color"] = "red"

      dot.node(n.getName(), n.getDotText(), attrs)

    for edge in self.getEdges():
      #print(edge)
      #print("DEBUG: adding edge to dot ", edge)
      #print("DEBUG: attrs=", edge.getAttrs())
      dot.edge(edge.getFromPosNode().node.getName(),
               edge.getToPosNode().node.getName(),
               edge.getLabel(), edge.getAttrs())

    # Add subgraphs
    clusters = {}
    for n in sorted(self.getNodes(), key=lambda x: x.getName()):
      # depth negative means to squash top level
      name = n.getName()
      if depth < 0:
        name = name.replace("/", "_", 1)
      clStrs = name.split("/")
      c = clusters
      #for i in range(0, len(clStrs)):
      for i in range(0, min(len(clStrs), abs(depth))):
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
    outFileAndExt = outFile + "." + outFormat
    print("INFO: invoking dot to render " + outFileAndExt)
    if MiscUtil.ExecTimeout.run(dot.render, outFile, Config.Dot.timeout):
      print("INFO: wrote " + outFileAndExt)
    else:
      print("INFO: dot rendering timed out, skipping")
      if os.path.exists(outFileAndExt):
        os.remove(outFileAndExt)

  # Finds all fanin nodes that are missing in this graph and copies
  # them as Constants. This eg, completes the boundary condition
  # for a subgraph that has all main flow nodes.
  # Returns list of input nodes
  def transferSideNodes(self, sourceGraph):
    inputNodes = []
    for n in self.getNodes():
      nodeName = n.getName()
      ns = sourceGraph.getNode(nodeName)
      assert ns != None
      for predNode in sourceGraph.nodePredecessors(ns):
        predName = predNode.getName()
        if self.debugLevel > 1:
          print("DEBUG: transferSideNodes %s -> %s, node type = %s" % (predName, nodeName, type(predNode)))
        if not self.hasNode(predName):

          if predNode.isMainFlowNode():
            predNodeCopy = predNode.copyAs(NodeInput, "Input")
            inputNodes.append(predNodeCopy)
            if self.debugLevel > 1:
              print("DEBUG: transferSideNodes copied node %s as Input" % (predName))
          else:
            predNodeCopy = predNode.copy()
            if self.debugLevel > 1:
              print("DEBUG: transferSideNodes copied node %s" % (predName))
          self.addNode(predNodeCopy)
          if self.debugLevel > 1:
            print("DEBUG: transferSideNodes added node %s" % (predNodeCopy.getName()))
          for srcEdge in predNode.getFanoutEdges():
            toName = srcEdge.getToNode().getName()
            if self.hasNode(toName):
              eNew = self.copyEdge(srcEdge)
              # Note - This is a noop since edge color is stored in attributes (not
              # recalculated from subgraph state). Coloring based on main graph
              # is perhaps even better
              #eNew.setIsInMainFlow(False)
    return list(set(inputNodes))

  # Returns file to append to the Kaena backend package
  def runScheduler(self, outPrefix):
    if self.schedulerMode == 'tcc':
      # Noop, wave scheduling is done in the backend
      return True, []
    elif self.schedulerMode == 'wave' or self.schedulerMode == 'wave2' or self.schedulerMode == 'qemu_wave' or self.schedulerMode == 'qemu_wave2':
      # Invoke wave scheduler
      waveSchedulerExec = self.kaenaPath + ("/compiler/me/me_main.py" if 'wave2' in self.schedulerMode else "/compiler/me/layeropt.py")
      kGraphJsonFile =  "compiler.json"
      waveGraphJsonFile = "wavegraph.json"

      # From Jeff: to generate dot without placement, but not svg:  waveDotFile = outPrefix + "wavegraph.plain"
      # From Jeff: to generate dot with placeemnt, but not svg:  waveDotFile = outPrefix + "wavegraph.dot"
      #waveDotFile = outPrefix + "wavegraph.svg"
      waveDotFile = outPrefix + "wavegraph.plain"
      fmt = "python3 %s %s --kgraph %s --wavegraph %s --dot %s --debug %d --verify_output_only > log-me.txt 2>&1"
      cmd = fmt % (waveSchedulerExec, Config.Scheduler.waveoptOptions, kGraphJsonFile, waveGraphJsonFile, waveDotFile, Config.debugLevel)

      print(getInfoDateStr(), "executing wave scheduler by  " + cmd)
      status = os.system(cmd)
      meOk = status == 0
      return meOk, [waveGraphJsonFile, waveDotFile]

  #def convertOpsWithNoArgsToConstNodes(self, OpTypes):
    #for n is self.getNodes():
    #  if n.getOpType in OpTypes:
    #    if len(n.getPredecessors()) == 0:
    #      nodeCopy = n.CopyAs(Node, "Const")
    #      nodeCopy.setLevel(n.getLevel())

  def optimizeForInference(self):
    pass
    # For a legacy resnet152
    #self.convertOpsWithNoArgsToConstNodes(["Multiply", "Sub"])

  def getOpsAndSizes(self, inputs, outputs):
    opCount = 0
    ifmapBytes = 0
    ofmapBytes = 0
    weightBytes = 0

    # Opcounts
    for n in self.getNodes():
      opCount += n.getOpCount()
    # Separate nodes into inputs, outputs, constants
    inputNodes = set(inputs)
    outputNodes = set(outputs)
    weightNodes = set([n for n in self.getNodes() if len(n.getFaninEdges()) == 0]) - inputNodes
    for n in inputNodes:
      ifmapBytes += n.getNpyInfoBytes()
    for n in outputNodes:
      ofmapBytes += n.getNpyInfoBytes()
    for n in weightNodes:
      weightBytes += n.getNpyInfoBytes()
    return opCount, weightBytes, ifmapBytes, ofmapBytes
