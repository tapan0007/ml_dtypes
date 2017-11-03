# Kaena abstraction of neural network framework operations
# Suitable for TF freeze graph input and general op reduction and fusing


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
    return("        # No model for %s %s\n" % (self.getOpType(), self.getOpName()))

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

class NodeConv2D(Node):
  def __init__(self, name, opType, strides, padding, dataFormat, attrs):
    Node.__init__(self, name, opType, attrs)
    self.__strides = strides
    self.__padding = padding
    self.__strides = strides
  def genCompilerLayerText(self):
    npInfo = self.getNpInfo()[0]
    s =  '        layer = ConvLayer(Layer.Param("%s", 1, self), layer,\n' % self.getOpName()
    s += '                   4, stride=1, kernel=3,\n'
    s += '                   filterFileName="%s", filterTensorDimSemantics="MCRS")\n' % "out_jdr_v3__weight1__read:0.npy"
    s += '        # Golden result file  %s\n' % npInfo.npFile
    s += "        \n"
    return(s)
    

# Computational data flow graph
class Graph(Object):
  def __init__(self, name, attrs = {}):
    Object.__init__(self, name, attrs)
    self.__name2node = {}
    self.__edges = []
    self.__mainFlowEdges = []
    
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


  def genCompilerPy(self, outFile):
    prefix = """
from utils.fmapdesc     import OfmapDesc
from utils.datatype     import *
 
from layers.layer       import Layer
from layers.datalayer   import DataLayer
from layers.convlayer   import ConvLayer

from nets.network       import Network
 
 
class TrivNet(Network):
    def __init__(self):
        super().__init__(DataTypeFloat16())

    def gName(self):
        return "TrivNet"
 
    def construct(self):
        layer =  DataLayer(Layer.Param("jdr_v3/input", 1, self),
                    OfmapDesc(3, 8), inputDataFileName="out_input:0.npy", dataTensorDimSemantics="NCHW")
 
"""
    lines = []
    levelizedNodes = self.getLevelizedNodes()
    for level in range(0, len(levelizedNodes)):
      for n in levelizedNodes[level]:
        op = n.getOpType()
        print("DEBUG: node=", n.getName(), "  op=", op)
        if op == "Conv2D":
          s = n.genCompilerLayerText()
          lines.append(s)

    with open(outFile, "w") as f:
      f.write(prefix)
      for s in lines:
        f.write(s)
    print("INFO: wrote ", outFile)
          
          
    
