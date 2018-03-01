# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Kaena abstraction of neural network framework operations
# Suitable for TF freeze graph input and general op reduction and fusing

from NpTransforms import NpTrans as npt
from NpUtils import NpUtils as npu
import os, re, json, sys
import numpy as np
import math
from graphviz import Digraph
sys.path.insert(0, os.environ["KAENA_PATH"] + "/compiler/tffe")
import MiscUtil

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
  class Dot:
    timeout = 60

class Object:
  def __init__(self, name, attrs):
    self.__name = name
    self.__attrs = attrs.copy()
  def getName(self):
    return(self.__name)
  def getAttr(self, attrName):
    return(self.__attrs[attrName])
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

# NN operation
class Node(Object):
  def __init__(self, name, opType, attrs):
    Object.__init__(self, name, attrs)
    self.__level = -1
    self.__opType = opType
    self.__npInfo = []
    self.__fanin = []  # inputs
    self.__fanout = [] # outputs
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
  def getFaninEdges(self):
    return(self.__fanin)
  def getFaninMainFlowEdges(self):
    return [e for e in self.getFaninEdges() if e.isInMainFlow()]
  def getFanoutEdges(self):
    return([item for edgelist in self.__fanout for item in edgelist])
  def getFanoutMainFlowEdges(self):
    return [e for e in self.getFanoutEdges() if e.isInMainFlow()]
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
  # Base class support for single-input, single output layers
  # E.g., activation, later possibly other simple layers
  # Returns list of layer maps (e.g., 2 for a node with a side constant),
  # and list of npy files (that need to be part of Kgraph package)
  def genCompilerLayerJson(self):
    return([{"layer_type" :  self.getOpType(),
            "layer_name" :  self.getOpName(),
            "#comment"   :  "unsupported layer"
            }], [])
  def getOpCount(self):
    return 1
  def getDotText(self):
    return self.getOpType()
  # Supported ops/nodes are passed down through the compiler and simulator flow
  def isSupported(self):
    return False
  def getOpArgsText(self):
    argsText = self.getDotText()
    # Simple implementation - reuse dot text and remove \n
    argsText = re.sub("\n", " ", argsText)
    return argsText
  def isMainFlowNode(self):
    return len(self.getFaninMainFlowEdges()) > 0 or len(self.getFanoutMainFlowEdges()) > 0

class PosNode:
  def __init__(self, node, index):
    self.node = node
    self.index = index
    
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
  def genCompilerLayerJson(self):
    fileList = []
    npInfo = self.getNpInfo()[0]
    if len(npInfo.npShape) == 4:
      tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
      (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    else:
      tfShape4D = npt.cShapeToNHWC(npInfo.npShape)
      (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D)
      tpbShape = list(npt.reorderShape(tfShape4D, npt.TF, npt.SIM, npt.Fmaps))
    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [],
      "#comment"        : "supported const layer"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True


###############################################################################
# Simple single input, single output nodes like RELU
###############################################################################
class NodeSimple(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self):
    fileList = []
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    ((fromIfNode, npInfoIF),) = self.getInputNodesAndNpInfo()
    (npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [fromIfNode.getName()],
      "#comment"        : "supported simple layer"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self)
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

      (HoCalc, padNorth, padSouth) = NodeConv2D.calcSamePadding(Sv, R, Hi)
      (WoCalc, padWest,  padEast)  = NodeConv2D.calcSamePadding(Sh, S, Wi)
      assert Ho == HoCalc
      assert Wo == WoCalc
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
  def genCompilerLayerJson(self):
    fileList = []
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW)) = self.getInputNodesAndNpInfo()
    tpbFilterShape = list(npt.reorderShape(npInfoW.npShape, npt.TF, npt.SIM, npt.Weights))
    # OFMAP
    (npFileSim, simFormatOF) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    # IFMAP, not needed
    (npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
    # WEIGHT
    (npFileSimW, simFormatW) = npt.copyNpyFileAs(npInfoW.npFile, npt.TF, npt.SIM, npt.Weights)

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
      "#comment"        : "supported layer"
    }
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  # Node text for dot graph
  def getDotText(self):
    dotText = self.getOpType()
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
      dotText += "\nGop %.3f" % (opCount / Config.Tpb.freq)
    return dotText

  # Number of add, multiply ops for performance analysis and reporting
  # E.g., 1 multiply and 1 accumulate is reported as 2 ops.
  def getOpCount(self):
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (batch, channels, height, width) = tpbShape
    ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW)) = self.getInputNodesAndNpInfo()
    filterShapeRSCM = npInfoW.npShape
    # 7 loops - 4 in the filter, 3 in ofmap
    opCount = 2 * np.empty(filterShapeRSCM).size * batch * height * width;
    return opCount
  

###############################################################################
# Max Pool
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
  def genCompilerLayerJson(self):
    fileList = []
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (batch, channels, height, width) = tpbShape
    (fromIfNode, npInfoIF) = self.getInputNodesAndNpInfo()[0]
    kernelSizeNHWC = self.getKernelSize()
    kernelSizeNCHW = [kernelSizeNHWC[i] for i in [0, 3, 1, 2]]
    # OFMAP
    (npFileSim, simFormatOF) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    # IFMAP, not needed
    (npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)

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
      "#comment"        : "supported layer"
    }
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self)
    layerDataBase[0].update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

  # Node text for dot graph
  def getDotText(self):
    dotText = self.getOpType()
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
    return dotText

  # Number of add, multiply, max or move, copy ops for performance
  # analysis and reporting
  # E.g., 1 multiply and 1 accumulate is reported as 2 ops.
  def getOpCount(self):
    npInfo = self.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (batch, channels, height, width) = tpbShape
    kernelSizeNHWC = self.getKernelSize()
    opCount = 2 * np.empty(kernelSizeNHWC).size * batch * height * width * channels;
    return opCount


###############################################################################
# Basic 2-argument operations (e.g., Mul, Add) with no config (e.g., filter,
# stride, padding)
###############################################################################
class NodeSimple2(Node):
  def __init__(self, name, opType, attrs):
    super().__init__(name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self):
    fileList = []
    
    # Output tensor
    npInfo = self.getNpInfo()[0]
    tfShape4D = npt.cShapeToNHWC(npInfo.npShape)
    tpbShape = list(npt.reorderShape(tfShape4D, npt.TF, npt.SIM, npt.Fmaps))
    (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D)
    
    # Residual Add has both inputs dependent on the input image
    # BiasAdd has the other input constant
    # In Keras plain Add can be of either of the above types
    (faninEdgeFmap, faninEdgeOther) = self.getFaninEdges()
    theOpIsInMainDataFlow = faninEdgeFmap.isInMainFlow()
    if not theOpIsInMainDataFlow:
      # This Add is a part of a side branch computation, no layer needed
      return [], []
    
    isResAdd = faninEdgeOther.isInMainFlow()
    ((fromIfNode0, npInfoIF0), (fromIfNode1, npInfoIF1),) = self.getInputNodesAndNpInfo()
    
    layerData = {
      "ofmap_shape"     : tpbShape,
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [fromIfNode0.getName(), fromIfNode1.getName()],
      "#comment"        : "supported simple layer with 2 inputs"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self)
    layerDataBase[0].update(layerData)
    fileListBase += fileList

    # Override layer name to backend
    #   BiasAdd - when one input is constant
    #   ResAdd - when both inputs depend on the input image
    overrideType = "BiasAdd"
    if isResAdd:
      overrideType = "ResAdd"
    layerDataBase[0]["layer_type"] = overrideType
       
    if not isResAdd and not type(fromIfNode1) == NodeConst:
      # Collapse the side node to a branch (except when it already is a real constant) 
      # Main input is covered by a previous layer
      #   tfShape4D0 = npt.cShapeToNHWC(npInfoIF0.npShape)
      #   (npFileSimF0, simFormatIF0)  = npt.copyNpyFileAs(npInfoIF0.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D0)
      # Side input has to be collapsed to a constant
      tfShape4D1 = npt.cShapeToNHWC(npInfoIF1.npShape)
      (npFileSimF1, simFormatIF1)  = npt.copyNpyFileAs(npInfoIF1.npFile, npt.TF, npt.SIM, npt.Fmaps, tfShape4D1)
      tpbShape4D1 = list(npt.reorderShape(tfShape4D1, npt.TF, npt.SIM, npt.Fmaps))
      
      constLayerData = {
       "layer_type" :  "Const",
       "layer_name" :  fromIfNode1.getName(),
        "ofmap_shape"     : tpbShape4D1,
        "ofmap_format"    : simFormat,
        "ref_file"        : npFileSimF1,
        "previous_layers" : [],
       "#comment"   :  "captured constant"
      }
      fileListBase.insert(0, npFileSimF1)
      layerDataBase.insert(0, constLayerData)  # prepend - because the backend Json parser requires layers defined

    return(layerDataBase, fileListBase)

  def isSupported(self):
    return True



###############################################################################
# Computational data flow graph
###############################################################################
class Graph(Object):
  def __init__(self, name = "GRAPH", attrs = {}, schedulerMode = "tcc"):
    super().__init__(name, attrs)
    self.__name2node = {}
    self.__edges = []
    self.__mainFlowEdges = []
    self.__inputNode = None
    self.kaenaPath = os.environ["KAENA_PATH"]
    self.schedulerMode = schedulerMode
    
  def setSchedulerMode(self, mode):
    self.schedulerMode = mode
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
  
  def nodeSuccessors(self, fromNode):
    nextNodes = {}
    for edge in fromNode.getFanoutEdges():
      nextNodes[edge.getToPosNode().node] = 1
    return(list(nextNodes.keys()))
  
  def nodePredecessors(self, toNode):
    preNodes = []
    for edge in toNode.getFaninEdges():
      preNodes.append(edge.getFromPosNode().node)
    return(preNodes)
  
  # Get the node with most computation - highest in the data flow level
  def getTopNode(self):
    nextNodes = self.getNodes()
    while len(nextNodes) > 0:
      n = nextNodes[0]
      nextNodes = self.nodeSuccessors(n)
    return(n)
  
  def setInputNode(self, node):
    self.__inputNode = node
  def getInputNode(self):
    return(self.__inputNode)
  
  # On a levelized graph - max depth to reach node among all paths
  # It describes "computational readiness" in the data flow:
  #   all ops at level N done implies that an op at level N+1 can
  #   be safely executed
  
  def levelize(self):
    nodes = self.getNodes()
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
    nodes = {self.getInputNode(): 0}
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
    print("INFO: executing  %s" %cmd)
    os.system(cmd)


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
    
    jsonData = {
      "net_name"  : "TrivNet",
      "layers"   : []
    }
       
    fileList = []

    # Input layer
    inputNode = self.getInputNode()
    npInfo = inputNode.getNpInfo()[0]
    tpbShape = list(npt.reorderShape(npInfo.npShape, npt.TF, npt.SIM, npt.Fmaps))
    (batch, channels, height, width) = tpbShape
    assert(height == width)
    jsonData["data_type"] = npInfo.dType   # No conversion by npu.dtypeToStr() was needed

    (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    inputLayerData = {
      "layer_name"      : inputNode.getName(),
      "layer_type"      : "Input",
      "previous_layers" : [],
      "ofmap_shape"     : [batch, channels, height, width],
      "ref_file"        : npFileSim,
      "ofmap_format"    : simFormat
    }
    jsonData["layers"].append(inputLayerData)
    fileList.append(npFileSim)
    outNpy = npFileSim
    if verbose > 0:
      npu.showNpyFile("Input IFMAPs", npFileSim)
    
    # Conv and other layers
    levelizedNodes = self.getLevelizedNodes()
    totalOpCount = 0
    for level in range(0, len(levelizedNodes)):
      for n in levelizedNodes[level]:
        if n == inputNode:
          continue
        op = n.getOpType()
        #print("DEBUG graph::genCompilerJson: node=", n.getName(), "  op=", op)
        #isMainFlowConst = (n.getOpType() == "Const" and any(ns.isMainFlowNode() for ns in self.nodeSuccessors(n)))
        #print("  DEBUG const and successor in main flow " + str(isMainFlowConst))
        if n.isSupported():
          (layerData, fileListLayer) = n.genCompilerLayerJson()
          if len(layerData) > 0:
            jsonData["layers"] += layerData
            fileList += fileListLayer
            outNpy = fileListLayer[-1]
        opCount = n.getOpCount()
        totalOpCount += opCount
        if Config.debugLevel >= 1:
          print("DEBUG: opcount is %d for %s  %s" % (opCount, n.getOpType(), n.getOpName()))
    print("INFO: total opcount is %d" % totalOpCount)
        

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
    
    return(outNpy, fileList)
    
  
  # Write K-graph compiler configuration files
  # Example:
  #  net_<interface_type>_params.sh : 
  #    OutputNpy=trivnet_1conv__i1:0_NCHW.npy
  #    PythonFile=trivnet_compiler.py
  def genKgraphSetupFiles(self, fileNamePy, fileNameJson, fileNameNpyOutput):
    #setupFilePy = "net_py_params.sh"
    #with open(setupFilePy, "w") as f:
    #  f.write("OutputNpy=%s\n" % fileNameNpyOutput)
    #  f.write("PythonFile=%s\n" % fileNamePy)
    setupFileJson = "net_json_params.sh"
    with open(setupFileJson, "w") as f:
      f.write("OutputNpy=%s\n" % fileNameNpyOutput)
      f.write("JsonFile=%s\n" % fileNameJson)
    return([setupFileJson])


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
        print("DEBUG: transferSideNodes %s -> %s" % (predName, nodeName))
        if not self.hasNode(predName):
          constNode = predNode.copyAs(NodeConst, "Const")
          self.addNode(constNode)
          print("DEBUG: transferSideNodes added node %s" % (constNode.getName()))
          for srcEdge in predNode.getFanoutEdges():
            toName = srcEdge.getToNode().getName()
            if self.hasNode(toName):
              eNew = self.copyEdge(srcEdge)
              # Note - This is a noop since edge color is stored in attributes (not
              # recalculated from subgraph state). Coloring based on main graph
              # is perhaps even better
              #eNew.setIsInMainFlow(False)
              if srcEdge.isInMainFlow():
                inputNodes.append(constNode)
    return list(set(inputNodes))

  # Returns file to append to the Kaena backend package
  def runScheduler(self, outPrefix):
    if self.schedulerMode == 'tcc':
      # Noop, wave scheduling is done in the backend
      return []
    elif self.schedulerMode == 'wave':
      # Invoke wave scheduler
      waveSchedulerExec = self.kaenaPath + "/compiler/util/layeropt.py"
      kGraphJsonFile = outPrefix + "compiler.json"
      waveGraphJsonFile = outPrefix + "wavegraph.json"
      waveDotFile = outPrefix + "wavegraph.svg"
      cmd = "python3 %s --kgraph %s --wavegraph %s --golden_inputs --debug %d > log-me.txt 2>&1" % (
            waveSchedulerExec, kGraphJsonFile, waveGraphJsonFile, Config.debugLevel)
      print("INFO: executing wave scheduler by  " + cmd)
      os.system(cmd)
      return [waveGraphJsonFile, waveDotFile]
    



