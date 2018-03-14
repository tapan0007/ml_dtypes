# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena operation graph (Kgraph) partitioning into smaller subgraphs.
# The partitioning has multiple goals, such as to run different parts
# of the NN graph on different backends, cut a small test case for debugging
# or regressions.

import os
import KaenaOpGraph as kog
import re
import shutil

# Describes a (small) Kgraph with inputs and the output node
class KsubGraph:
  def __init__(self, debugLevel):
    self.graph = kog.Graph()
    self.__inputs = []
    self.__output = None
    self.__maxLevel = 0    # highest level of any node (== output) in the src graph
    self.debugLevel = debugLevel
    self.isSupported = True
  def print(self, title):
    print(title)
    self.graph.print()
  def getMaxLevel(self):
    return self.__maxLevel
  def updateMaxLevel(self, level):
    self.__maxLevel = max(self.__maxLevel, level)
  # Links the npyinfo files used by the graph from the srcDir
  def relinkNpFiles(self, srcDir):
    for n in self.graph.getNodes():
      for npInfo in n.getNpInfo():
        f = npInfo.npFile
        os.symlink("%s/%s" % (srcDir, f), f)
  def genExecutorGraphJson(self, sgDir):
    jsonDict = {"SubGraphDir" : sgDir}
    jsonDict["Inputs"] = []
    for ni in self.__inputs:
      inNpFile = ni.getNpInfo()[0].npFile
      inShape = ni.getNpInfo()[0].npShape
      jsonDict["Inputs"] = [{"name" : ni.getName() + ":0", "file" : inNpFile, "shape" : inShape}]
    o = self.__output
    outNpFile = o.getNpInfo()[0].npFile
    outShape = o.getNpInfo()[0].npShape
    jsonDict["Outputs"] = [{"name" : o.getName() + ":0", "file" : outNpFile, "shape" : outShape}]
    return jsonDict
  def addSideNodes(self, srcGraph):
    self.__inputs = self.graph.transferSideNodes(srcGraph)
    self.__output = self.graph.getTopNode()
    # In the very first subgraph the original input node needs to be added too
    srcInputName = srcGraph.getInputNode().getName()
    if self.graph.hasNode(srcInputName):
      srcInpEqNode = self.graph.getNode(srcInputName)
      if not srcInpEqNode in self.__inputs:
        self.__inputs.insert(0, srcInpEqNode)
    if len(self.__inputs) == 0:
      self.__inputs.append(self.__output)
    # Make one of the nodes input for the backend, should not matter which one
    inputNode = self.__inputs[0]
    if self.debugLevel > 0:
      print("DEBUG: set input node %s" % inputNode.getName())
    self.graph.setInputNode(inputNode)
    #hasInputOpType = any((ni.getOpType() == "Input") for ni in self.__inputs)
    #assert inputNode.getOpType() == "Input" or
    #       inputNode.getOpType() == "Const"
  def addNode(self, n):
    self.graph.addNode(n)
    if not n.isSupported():
      self.isSupported = False

  
# Graph partitioner
class KgraphPart(object):

  def __init__(self, kgraph, debugLevel):
    self.__kgraph = kgraph
    self.__subgraphs = []
    self.__node2color = {}
    self.__numColors = 0
    self.sgId2executor = {}
    self.debugLevel = debugLevel
  
  def getNewColor(self):
    color = self.__numColors
    self.__numColors += 1
    return(color)
  
  def setNodeColor(self, node,color):
    self.__node2color[node] = color

  def getNodeColor(self, node):
    return self.__node2color.get(node, None)

  def getSubgraphs(self):
    return self.__subgraphs
  
  # Returns the predecessor nodes along main flow edges only
  def getPredecessorMainFlowNodes(self, node):
    faninEdges = node.getFaninMainFlowEdges()
    predNodes = [e.getFromNode() for e in faninEdges]
    return(predNodes)

  # Returns the node, asserts that there is exactly one
  def getPredecessorMainFlowNode(self, node):
    predNodes = self.getPredecessorMainFlowNodes(node)
    assert len(predNodes) == 1
    return(predNodes[0])

  # Returns True if nodes predecessor is forking flow (fanout >= 2)
  def predNodeHasFanout(self, node):
    predNode = self.getPredecessorMainFlowNode(node)
    predFanout = len(predNode.getFanoutMainFlowEdges())
    return(predFanout > 1)
  
  # Auto color nodes to define subgraph partitions
  def colorNodesAuto(self):
    sourceGraph = self.__kgraph
    edgeQueue = []
    visitedNodes = {}
    n = sourceGraph.getInputNode()
    self.setNodeColor(n, self.getNewColor())
    edgeQueue += n.getFanoutMainFlowEdges()
    while len(edgeQueue) > 0:
      e = edgeQueue.pop(0)
      n = e.getToNode()
      if n in visitedNodes:
        continue
      visitedNodes[n] = True
      print("DEBUG: colorNodesAuto visit         %-12s %s" %
              (n.getOpType(), n.getName()))
      fanoutEdges = n.getFanoutMainFlowEdges()
      faninEdges = n.getFaninMainFlowEdges()
      # Depth-first traversal upto any reconvergence point
      if len(faninEdges) > 1:
        edgeQueue += fanoutEdges
      else:
        if len(fanoutEdges) > 0:
          edgeQueue.insert(0, fanoutEdges[0])
          edgeQueue += fanoutEdges[1:]
      # Coloring
      if len(faninEdges) > 1:
        # Reconvergence node is standalone partition
        self.setNodeColor(n, self.getNewColor())
      elif self.predNodeHasFanout(n):
        self.setNodeColor(n, self.getNewColor())
      else:
        assert len(faninEdges) == 1
        # Fanout node stays with the previous partition
        predNode = self.getPredecessorMainFlowNode(n)
        self.setNodeColor(n, self.getNodeColor(predNode))
      if self.debugLevel > 0:
        print("DEBUG: colorNodesAuto setColor %d on %-12s %s" %
              (self.getNodeColor(n), n.getOpType(), n.getName()))
  
  # Auto color nodes to define subgraph partitions supported by the middle-end
  # The partitions end on multi fanout node. Every graph cut has exactly 1 node.
  # This is done by simple traversals till node with fanout
  def colorNodesMeAuto(self):
    sourceGraph = self.__kgraph
    edgeQueue = []
    visitedNodes = set()
    n = sourceGraph.getInputNode()
    color = self.getNewColor()
    self.setNodeColor(n, color)
    for e in n.getFanoutMainFlowEdges():
      edgeQueue.append((color, e))
    while len(edgeQueue) > 0:
      color, e = edgeQueue.pop(0)
      n = e.getToNode()
      if n in visitedNodes:
        continue
      visitedNodes.add(n)
      print("DEBUG: colorNodesMeAuto visit         %-12s %s" %
              (n.getOpType(), n.getName()))
      fanoutEdges = n.getFanoutMainFlowEdges()
      faninEdges = n.getFaninMainFlowEdges()
      self.setNodeColor(n, color)
      if color + 1 > self.__numColors:
        self.__numColors = color + 1
      print("DEBUG: colorNodesMeAuto setColor %d on %-12s %s" %
            (self.getNodeColor(n), n.getOpType(), n.getName()))
      # Color of the next partition
      nextColor = color
      if len(fanoutEdges) > 1:
        nextColor = color + 1
      # Traverse deeper
      for nextEdge in fanoutEdges:
        edgeQueue.append((nextColor, nextEdge))


  # Auto color nodes to define partitions based on supported nodes.
  def colorNodesSuppAuto(self):
    sourceGraph = self.__kgraph
    edgeQueue = []
    visitedNodes = {}
    n = sourceGraph.getInputNode()
    self.setNodeColor(n, self.getNewColor())
    edgeQueue += n.getFanoutMainFlowEdges()
    while len(edgeQueue) > 0:
      e = edgeQueue.pop(0)
      n = e.getToNode()
      p = e.getFromNode()
      edgeQueue += n.getFanoutMainFlowEdges()
      if n in visitedNodes:
        continue
      visitedNodes[n] = True
      print("DEBUG: colorSuppAuto visit         %-12s %s" %
            (n.getOpType(), n.getName()))

      if n.isSupported() != p.isSupported():
        print("DEBUG: adding new color for node: " +
              n.getName() +
              " " + str(n.isSupported()) +
              " predNode " + p.getName() +
              " " + str(p.isSupported()) )
        self.setNodeColor(n, self.getNewColor())
      else:
        self.setNodeColor(n, self.getNodeColor(p))

      if self.debugLevel > 0:
        print("DEBUG: colorSuppAuto setColor %d on %-12s %s" %
              (self.getNodeColor(n), n.getOpType(), n.getName()))


  def edgeHasOp(self, edge, opType):
    nodes = [edge.getFromNode(), edge.getToNode()]
    return any((not n == None and n.getOpType() == opType) for n in nodes)
  
  # Color nodes to define subgraph partitions -isolate conv2d
  def colorNodesConv(self):
    sourceGraph = self.__kgraph
    edgeQueue = []
    visitedNodes = {}
    n = sourceGraph.getInputNode()
    self.setNodeColor(n, self.getNewColor())
    edgeQueue += n.getFanoutMainFlowEdges()
    while len(edgeQueue) > 0:
      e = edgeQueue.pop(0)
      n = e.getToNode()
      if n in visitedNodes:
        continue
      visitedNodes[n] = True
      if self.debugLevel > 0:
        print("DEBUG: colorNodesConv visit         %-12s %s" %
              (n.getOpType(), n.getName()))
      fanoutEdges = n.getFanoutMainFlowEdges()
      faninEdges = n.getFaninMainFlowEdges()
      # Depth-first traversal upto any reconvergence point
      if len(faninEdges) > 1:
        edgeQueue += fanoutEdges
      else:
        if len(fanoutEdges) > 0:
          edgeQueue.insert(0, fanoutEdges[0])
          edgeQueue += fanoutEdges[1:]
      # Coloring
      if len(faninEdges) > 1:
        # Reconvergence node is standalone partition
        self.setNodeColor(n, self.getNewColor())
      elif self.predNodeHasFanout(n):
        self.setNodeColor(n, self.getNewColor())
      elif self.edgeHasOp(e, "Conv2D"):
        self.setNodeColor(n, self.getNewColor())
      else:
        assert len(faninEdges) == 1
        # Fanout node stays with the previous partition
        predNode = self.getPredecessorMainFlowNode(n)
        self.setNodeColor(n, self.getNodeColor(predNode))
      if self.debugLevel > 0:
        print("DEBUG: colorNodesConv setColor %d on %-12s %s" %
              (self.getNodeColor(n), n.getOpType(), n.getName()))

  # Color nodes to define subgraph partitions - start new partition from a node
  # The from list must NOT be part of any reconvergent branch
  def colorNodesFrom(self, fromNodeList):
    sourceGraph = self.__kgraph
    fromNodeSet = set(fromNodeList)
    edgeQueue = []
    visitedNodes = {}
    n = sourceGraph.getInputNode()
    color = self.getNewColor()
    self.setNodeColor(n, color)
    edgeQueue += [(e, color) for e in n.getFanoutMainFlowEdges()]
    while len(edgeQueue) > 0:
      e,color = edgeQueue.pop(0)
      n = e.getToNode()
      if n in visitedNodes:
        continue
      visitedNodes[n] = True
      if self.debugLevel > 0:
        print("DEBUG: colorNodesFrom visit         %-12s %s" %
              (n.getOpType(), n.getName()))
      # Coloring
      if n.getName() in fromNodeSet:
        color = self.getNewColor()
        self.setNodeColor(n, color)
      else:
        self.setNodeColor(n, color)
      fanoutEdges = n.getFanoutMainFlowEdges()
      edgeQueue += [(e, color) for e in fanoutEdges]
      if self.debugLevel > 0:
        print("DEBUG: colorNodesFrom setColor %d on %-12s %s" %
              (self.getNodeColor(n), n.getOpType(), n.getName()))

  # Color nodes given the partitioning strategy
  # The strategy is a keyword and arguments (for some)
  def colorNodes(self, partitioningStrategy):
    strategy = partitioningStrategy[0]
    if strategy == "auto":
      self.colorNodesAuto()
    elif strategy == "meauto":
      self.colorNodesMeAuto()
    elif strategy == "conv":
      self.colorNodesConv()
    elif strategy == "suppauto":
      self.colorNodesSuppAuto()
    elif strategy == "from":
      self.colorNodesFrom( partitioningStrategy[1:])
    else:
      assert 0

  # Partition into ordered list of subgraphs
  # All partitions have single output, multiple inputs
  def partitionByColor(self):
    sourceGraph = self.__kgraph
    for i in range(self.__numColors):
      self.__subgraphs.append(KsubGraph(self.debugLevel))
    levelizedNodes = sourceGraph.getLevelizedNodes()
    # Nodes
    for level in range(len(levelizedNodes)):
      for n in levelizedNodes[level]:
        color = self.getNodeColor(n)
        if color != None:
          subGraph = self.__subgraphs[color]
          nCopy = n.copy()
          assert(not n.getFaninEdges == None)
          subGraph.addNode(nCopy)
          subGraph.updateMaxLevel(level)
    # Edges
    for i in range(self.__numColors):
      sg = self.__subgraphs[i]
      #sg.print("Subgraph %d" % i)
      sg.graph.copyEdges(sourceGraph)
      #for e in sg.graph.getEdges():
      #  e.setIsInMainFlow(True)
    # Side nodes
    # to add
    
    # Levelize subgraphs - done in tffe to make flow more serial per sg
    #for sg in self.getSubgraphs():
    #  sg.print("Subgraph pre-levelize")
    #  sg.graph.levelize()
    
    # Order subgraphs that runtime dependencies are satisfied
    # Simple sorting by the level of the output node is enough
    self.__subgraphs.sort(key = lambda sg : sg.getMaxLevel())

  # Print textual connectivity info to STDOUT
  def print(self):
    for i in range(self.__numColors):
      sg = self.__subgraphs[i]
      sg.print("Subgraph %d" % i)
  
  # Note nn_executor has a similar function, sharing compiler-runtime is not desirable
  def calcExecutorMap(self, executorsList):
    self.sgId2executor = {}
    
    # setup default executor: host if SG unsupported
    for sgId in range(len(self.__subgraphs)):
      if self.__subgraphs[sgId].isSupported:
        self.sgId2executor[sgId] = "tcc"
      else:
        self.sgId2executor[sgId] = "host"

    executor = None
    for word in executorsList:
      if word == "all":
        for sgId in range(len(self.__subgraphs)):
          self.sgId2executor[sgId] = executor
      elif re.search('^\d+$', word):
        sgId = int(word)
        self.sgId2executor[sgId] = executor
      else:
        executor = word
    print("INFO: subgraph to executor map  %s" % str(self.sgId2executor), flush=True)
    #assert len(self.sgId2executor) == len(self.__subgraphs)
  
  def getExecutorById(self, sgId):
    return self.sgId2executor.get(sgId, 'host')

def attachPrePost(sgJsonList, preprocessor, postprocessor):
  for (sgname) in ["sg_pre", "sg_post"]:
    if sgname == "sg_pre":
      f = preprocessor
    else:
      f = postprocessor
    if f != "":
      sgDir = sgname
      print("\nINFO: processing subgraph %s" % sgDir)
      os.makedirs(sgname)
      assert(os.path.isfile(f) and os.access(f, os.X_OK))
      shutil.copy2(f, os.getcwd() + "/" + sgname)
      sgJson = {}
      sgJson["executor"] = "processor"
      sgJson["SubGraphDir"] = sgname
      sgJson["cmd"] =  os.path.basename(f)
      sgJson["Inputs"] = []
      sgJson["Outputs"] = []
      if sgname == "sg_pre":
        sgJson["Outputs"] += sgJsonList[0]["Inputs"]
        sgJsonList.insert(0, sgJson)
      else:
        sgJson["Inputs"] += sgJsonList[-1]["Outputs"]
        sgJsonList.append(sgJson)











