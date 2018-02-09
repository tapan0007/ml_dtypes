# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Kaena operation graph (Kgraph) partitioning into smaller subgraphs.
# The partitioning has multiple goals, such as to run different parts
# of the NN graph on different backends, cut a small test case for debugging
# or regressions.


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
  def autoColorNodes(self):
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
        print("DEBUG: autoColorNodes visit         %-12s %s" %
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
        print("DEBUG: autoColorNodes setColor %d on %-12s %s" %
              (self.getNodeColor(n), n.getOpType(), n.getName()))
      

  # Partition into ordered list of subgraphs
  # All partitions have single output, multiple inputs
  def partitionByColor(self):
    sourceGraph = self.__kgraph
    for i in range(self.__numColors):
      self.__subgraphs.append(KsubGraph())
    levelizedNodes = sourceGraph.getLevelizedNodes()
    for level in range(len(levelizedNodes)):
      for n in levelizedNodes[level]:
        color = self.getNodeColor(n)
        if color != None:
          subGraph = self.__subgraphs[color]
          subGraph.graph.addNode(n)
  
  def print(self):
    for i in range(self.__numColors):
      sg = self.__subgraphs[i]
      sg.print("Subgraph %d" % i)
        
            
      
















