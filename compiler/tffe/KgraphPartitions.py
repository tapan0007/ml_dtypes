# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena operation graph (Kgraph) partitioning into smaller subgraphs.
# The partitioning has multiple goals, such as to run different parts
# of the NN graph on different backends, cut a small test case for debugging
# or regressions.

import os
import KaenaOpGraph as kog


# Describes a (small) Kgraph with names of inputs and name of the output
class KsubGraph:
  def __init__(self):
    self.graph = kog.Graph()
    self.__inputs = []
    self.__output = None
  def print(self, title):
    print(title)
    self.graph.print()
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
      #inp = {"Node" : ni.getName(), "NpFile" : ni.getNpInfo()[0].npFile}
      #jsonDict["Inputs"].append(inp)
      jsonDict["Inputs"].append(ni.getName() + ":0")
      jsonDict["Inputs"].append(ni.getNpInfo()[0].npFile)
    o = self.__output
    #out = {"Node" : o.getName(), "NpFile" : o.getNpInfo()[0].npFile}
    outNpFile = o.getNpInfo()[0].npFile
    outNpFile = outNpFile[:-4] + "-out.npy"
    jsonDict["Output"] = [o.getName() + ":0", outNpFile]
    return jsonDict
  def addSideNodes(self, srcGraph):
    self.__inputs = self.graph.transferSideNodes(srcGraph)
    self.__output = self.graph.getTopNode()
    if len(self.__inputs) == 0:
      self.__inputs.append(self.__output)
    # Make one of the nodes input for teh backend, should not matter whichone
    inputNode = self.__inputs[0]
    self.graph.setInputNode(inputNode)
    #hasInputOpType = any((ni.getOpType() == "Input") for ni in self.__inputs)
    #assert inputNode.getOpType() == "Input" or
    #       inputNode.getOpType() == "Const"
    
# Graph partitioner
class KgraphPart(object):

  def __init__(self, kgraph, debugLevel):
    self.__kgraph = kgraph
    self.__subgraphs = []
    self.__node2color = {}
    self.__numColors = 0
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
  
  # Returns the node, asserts that there is exactly one
  def getPredecessorMainFlowNode(self, node):
    faninEdges = node.getFaninMainFlowEdges()
    assert len(faninEdges) == 1
    predNode = faninEdges[0].getFromNode()
    return(predNode)

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
      if self.debugLevel > 0:
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
      
  # Color nodes given the partitioning startegy
  def colorNodes(self, partitioningStrategy):
    if partitioningStrategy == "auto":
      self.colorNodesAuto()
    elif partitioningStrategy == "conv":
      self.colorNodesConv()
    else:
      assert 0

  # Partition into ordered list of subgraphs
  # All partitions have single output, multiple inputs
  def partitionByColor(self):
    sourceGraph = self.__kgraph
    for i in range(self.__numColors):
      self.__subgraphs.append(KsubGraph())
    levelizedNodes = sourceGraph.getLevelizedNodes()
    # Nodes
    for level in range(len(levelizedNodes)):
      for n in levelizedNodes[level]:
        color = self.getNodeColor(n)
        if color != None:
          subGraph = self.__subgraphs[color]
          nCopy = n.copy()
          subGraph.graph.addNode(nCopy)
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
  
  # Print textual connectivity info to STDOUT
  def print(self):
    for i in range(self.__numColors):
      sg = self.__subgraphs[i]
      sg.print("Subgraph %d" % i)
  















