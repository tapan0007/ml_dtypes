# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Kaena abstraction of neural network framework operations
# Suitable for TF freeze graph input and general op reduction and fusing

from NpTransforms import NpTrans as npt
from NpUtils import NpUtils as npu
import os, re, json

class Object():
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

# NN operation
class Node(Object):
  def __init__(self, name, opType, attrs):
    Object.__init__(self, name, attrs)
    self.__level = -1
    self.__opType = opType
    self.__npInfo = []
    self.__fanin = []  # inputs
    self.__fanout = [] # outputs
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
  def getFanoutEdges(self):
    return([item for edgelist in self.__fanout for item in edgelist])
  # Fanin of 1 per input
  def setFaninEdge(self, edge, index):
    assert(len(self.__fanin) < index + 1 or self.__fanout[index] == None)
    while len(self.__fanin) < index + 1:
      self.__fanin.append(None)
    self.__fanin[index] = edge
  # Fanout can be greater than 1
  def setFanoutEdge(self, edge, index):
    while len(self.__fanout) < index + 1:
      self.__fanout.append([])
    self.__fanout[index].append(edge)
  def genCompilerLayerText(self):
    return("        # No model for %s %s\n" % (self.getOpType(), self.getOpName()), [])
  # Base class support for single-input, single output layers
  # E.g., activation, later possibly other simple layers
  def genCompilerLayerJson(self):
    return({"layer_type" :  self.getOpType(),
            "layer_name" :  self.getOpName(),
            "#comment"   :  "unsupported layer"
            }, [])

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
  def getFromPosNode(self):
    return(self.__fromPosNode)
  def getToPosNode(self):
    return(self.__toPosNode)
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

# Simple single input, single output nodes like RELU
class NodeSimple(Node):
  def __init__(self, name, opType, attrs):
    Node.__init__(self, name, opType, attrs)

  # Returns layer json model in dictionary format, and list of files (npy data)
  def genCompilerLayerJson(self):
    fileList = []
    npInfo = self.getNpInfo()[0]
    (batch, height, width, channels) = npInfo.npShape # FIX_THIS - this order is TF specific, use NpUtils
    (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    ((fromIfNode, npInfoIF),) = self.getInputNodesAndNpInfo()
    (npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
    layerData = {
      "ofmap_shape"     : [batch, channels, height, width],
      "ofmap_format"    : simFormat,
      "ref_file"        : npFileSim,
      "previous_layers" : [fromIfNode.getName()],
      "#comment"        : "supported simple layer"
    }
    fileList.append(npFileSim)
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self)
    layerDataBase.update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

class NodeConv2D(Node):
  def __init__(self, name, opType, padding, dataFormat, attrs):
    Node.__init__(self, name, opType, attrs)
    self.__padding = padding
    self.__dataFormat = dataFormat

  def getStrides(self):
    return(self.getAttr("strides"))
  
  # Returns layer python model in text format, and list of files (npy data)
  def genCompilerLayerText(self):
    fileList = []
    npInfo = self.getNpInfo()[0]
    (batch, height, width, channels) = npInfo.npShape
    ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW)) = self.getInputNodesAndNpInfo()
    filterSize = npInfoW.npShape[0]
    assert(npInfoW.npShape[0] == npInfoW.npShape[1]) # square filter
    # OFMAP
    (npFileSim, simFormatOF) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    # IFMAP, not needed
    (npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
    # WEIGHT
    (npFileSimW, simFormatW) = npt.copyNpyFileAs(npInfoW.npFile, npt.TF, npt.SIM, npt.Weights)
    stride = 1 # TO_DO extract it properly
    s =  '        layer = ConvLayer(Layer.Param("%s", %d, self), layer,\n' % (self.getOpName(), batch)
    s += '                   %d, stride=%d, kernel=%d,\n' % (channels, stride, filterSize)
    s += '                   filterFileName="%s", filterTensorDimSemantics="%s")\n' % (npFileSimW, simFormatW)
    s += '        # Golden result file  %s\n' % npFileSim
    s += "        \n"
    fileList += [npFileSimW, npFileSim]
    return(s, fileList)

  # Returns layer json model in dictionary format, and list of files (npy data)
  # To be removed once full flow is tested with JSON
  def genCompilerLayerJson(self):
    fileList = []
    npInfo = self.getNpInfo()[0]
    (batch, height, width, channels) = npInfo.npShape
    ((fromIfNode, npInfoIF), (fromWeightNode, npInfoW)) = self.getInputNodesAndNpInfo()
    filterShapeRSCM = npInfoW.npShape
    filterShapeMCRS = [filterShapeRSCM[i] for i in [3, 2, 0, 1]]
    # OFMAP
    (npFileSim, simFormatOF) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    # IFMAP, not needed
    (npFileSimF, simFormatIF)  = npt.copyNpyFileAs(npInfoIF.npFile, npt.TF, npt.SIM, npt.Fmaps)
    # WEIGHT
    (npFileSimW, simFormatW) = npt.copyNpyFileAs(npInfoW.npFile, npt.TF, npt.SIM, npt.Weights)

    fileList += [npFileSimW, npFileSim]
    stride = npt.reorderShape(self.getStrides(), npt.TF, npt.SIM, npt.Fmaps)
    padding = [[0,0], [0,0], [1,1], [1,1]]   # TO_DO extract it properly
    layerData = {
      "layer_type"      : "Conv",
      "kernel_file"     : npFileSimW,
      "kernel_format"   : simFormatW,
      "kernel_shape"    : filterShapeMCRS,
      "ofmap_shape"     : [batch, channels, height, width],
      "ofmap_format"    : simFormatOF,
      "ref_file"        : npFileSim,
      "padding"         : padding,
      "previous_layers" : [fromIfNode.getName()],
      "stride"          : stride,
      "#comment"        : "supported layer"
    }
    (layerDataBase, fileListBase) = Node.genCompilerLayerJson(self)
    layerDataBase.update(layerData)
    fileListBase += fileList
    return(layerDataBase, fileListBase)

# Computational data flow graph
class Graph(Object):
  def __init__(self, name, attrs = {}):
    Object.__init__(self, name, attrs)
    self.__name2node = {}
    self.__edges = []
    self.__mainFlowEdges = []
    self.__inputNode = None
    
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
    return(nextNodes.keys())
  
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
  def identifyMainFlowEdges(self, inputTensorName):
    nodes = {self.getNode(inputTensorName): 0}
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
  
  def edgeIsInMainFlow(self, edge):
    return(edge in self.__mainFlowEdges)


  # Generate Kaena Compiler input graph (in python format)
  # including weights, input and golden output imges (in numpy format)
  #
  # NPY file formats:
  #   source (in TF) -> conversions -> NPY written by TFFE
  # 
  # IFMAPS:
  #   NHWC  swapaxes(1,2)  NWHC   (1,3) NCHW 
  # 
  # WEIGHTS:
  #   RSCM   (0, 3)  MSCR   (1,2)  MCSR   (2,3)  MCRS
    
  def genCompilerPy(self, outFile, verbose):
    
    prefix = """
from utils.fmapdesc     import OfmapDesc
from utils.datatype     import *
 
from layers.layer       import Layer
from layers.datalayer   import DataLayer
from layers.convlayer   import ConvLayer

from nets.network       import Network
 
 
class TrivNet(Network):
    def __init__(self):
        super().__init__(DataTypeFloat16(), "TrivNet")

    def construct(self):
"""
    lines = []
    
    fileList = []

    # Input layer
    inputNode = self.getInputNode()
    npInfo = inputNode.getNpInfo()[0]
    (batch, height, width, channels) = npInfo.npShape
    assert(height == width)
    (npFileSim, simFormat) = npt.copyNpyFileAs(npInfo.npFile, npt.TF, npt.SIM, npt.Fmaps)
    s = '        layer =  DataLayer(Layer.Param("%s", %d, self),\n' % (inputNode.getOpName(), batch)
    s+= '               OfmapDesc(%d, %d), inputDataFileName="%s", dataTensorDimSemantics="%s")\n' % (channels, height, npFileSim, simFormat)
    lines.append(s)
    if verbose > 0:
      npu.showNpyFile("Input IFMAPs", npFileSim)
    
    # Conv and other layers
    levelizedNodes = self.getLevelizedNodes()
    outNpy = None
    for level in range(0, len(levelizedNodes)):
      for n in levelizedNodes[level]:
        op = n.getOpType()
        #print("DEBUG: node=", n.getName(), "  op=", op)
        if op == "Conv2D":
          (s, fileListLayer) = n.genCompilerLayerText()
          lines.append(s)
          fileList += fileListLayer
          outNpy = fileListLayer[-1]

    if verbose > 0:
      npu.showNpyFile("Output OFMAPs", outNpy)

    with open(outFile, "w") as f:
      f.write(prefix)
      for s in lines:
        f.write(s)
    print("INFO: wrote ", outFile)
    fileList.append(outFile)
    return(fileList)
    
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
      "data_type" : "float16",
      "layers"   : []
    }
       
    fileList = []

    # Input layer
    inputNode = self.getInputNode()
    npInfo = inputNode.getNpInfo()[0]
    (batch, height, width, channels) = npInfo.npShape
    assert(height == width)
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
    if verbose > 0:
      npu.showNpyFile("Input IFMAPs", npFileSim)
    
    # Conv and other layers
    levelizedNodes = self.getLevelizedNodes()
    outNpy = None
    for level in range(0, len(levelizedNodes)):
      for n in levelizedNodes[level]:
        op = n.getOpType()
        #print("DEBUG: node=", n.getName(), "  op=", op)
        if op == "Conv2D" or op == "Relu" or op == "Tanh":
          (layerData, fileListLayer) = n.genCompilerLayerJson()
          jsonData["layers"].append(layerData)
          fileList += fileListLayer
          outNpy = fileListLayer[-1]

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
    setupFilePy = "net_py_params.sh"
    with open(setupFilePy, "w") as f:
      f.write("OutputNpy=%s\n" % fileNameNpyOutput)
      f.write("PythonFile=%s\n" % fileNamePy)
    setupFileJson = "net_json_params.sh"
    with open(setupFileJson, "w") as f:
      f.write("OutputNpy=%s\n" % fileNameNpyOutput)
      f.write("JsonFile=%s\n" % fileNameJson)
    return([setupFilePy, setupFileJson])


