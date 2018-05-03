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
import numpy as np

# Describes a (small) Kgraph with inputs and the output node
class KsubGraph:
  def __init__(self, debugLevel):
    self.graph = kog.Graph(debugLevel=debugLevel)
    self.__inputs = []
    self.__outputs = []
    self.__maxLevel = 0    # highest level of any node (== output) in the src graph
    self.debugLevel = debugLevel
    self.isSupported = True
  def print(self, title):
    print(title)
    self.graph.print()
  def getInputs(self):
    return self.__inputs
  def getOutputs(self):
    return self.__outputs
  def getMaxLevel(self):
    return self.__maxLevel
  def updateMaxLevel(self, level):
    self.__maxLevel = max(self.__maxLevel, level)
  def addOutput(self, node):
    self.__outputs.append(node)
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
      jsonDict["Inputs"].append({"name" : ni.getName() + ":0", "file" : inNpFile, "shape" : inShape})
    jsonDict["Outputs"] = []
    for no in self.__outputs:
      outNpFile = no.getNpInfo()[0].npFile
      outShape = no.getNpInfo()[0].npShape
      jsonDict["Outputs"].append({"name" : no.getName() + ":0", "file" : outNpFile, "shape" : outShape})
    return jsonDict
  def addSideNodes(self, srcGraph):
    self.__inputs = self.graph.transferSideNodes(srcGraph)
    #self.__outputs = self.graph.getTopNodes()
    # In the very first subgraph the original input node needs to be added too
    srcInputName = srcGraph.getInputNode().getName()
    if self.graph.hasNode(srcInputName):
      srcInpEqNode = self.graph.getNode(srcInputName)
      if not srcInpEqNode in self.__inputs:
        self.__inputs.insert(0, srcInpEqNode)
    if len(self.__inputs) == 0:
      self.__inputs += self.__outputs
    # Make one of the nodes input for the backend, should not matter which one
    inputNodes = self.__inputs
    if self.debugLevel > 0:
      print("DEBUG: set input nodes %s" % str([n.getName() for n in inputNodes]))
    self.graph.setInputNodes(inputNodes)
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
  
  # Returns the successor nodes along main flow edges only
  def getSuccessorMainFlowNodes(self, node):
    fanoutEdges = node.getFanoutMainFlowEdges()
    succNodes = [e.getToNode() for e in fanoutEdges]
    return succNodes

  # Returns True if the node has at least one successor with a different color or no fanout
  def nodeIsSgOutput(self, node):
    fanoutEdges = node.getFanoutMainFlowEdges()
    if len(fanoutEdges) == 0:
      return True
    else:
      succNodes = [e.getToNode() for e in fanoutEdges]
      succColors = [self.getNodeColor(n) for n in succNodes]
      nodeColor = self.getNodeColor(node)
      return any(c != nodeColor for c in succColors)

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
      if self.debugLevel > 0:
        print("DEBUG: colorSuppAuto visit         %-12s %s" %
              (n.getOpType(), n.getName()))

      if n.isSupported() != p.isSupported():
        if self.debugLevel > 0:
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

  # Levelized (bi) coloring based on supported nodes. It should replace colorNodesSuppAuto
  # FIX_THIS: this is partial code; needs to have auto-detection for first partion type added
  # to avoid generating empty subgraph
  def colorNodesLevelAuto(self):
    sourceGraph = self.__kgraph
    edge2color = {}
    visitedNodes = set()
    nodeFront = sourceGraph.getInputNodes()
    color = self.getNewColor()
    sgSupported = True
    for n in nodeFront:
      self.setNodeColor(n, color)
      if self.debugLevel > 0:
        print("DEBUG: colorNodesLevelAuto INPUTS setColor %d on %-12s %s" %
              (self.getNodeColor(n), n.getOpType(), n.getName()))

    newNodeFrontNext = [None]
    # Black/white loop
    while len(newNodeFrontNext) > 0:
      newNodeFrontNext = []
      change = True
      # Same color expansion loop
      while change:
        change = False
        newNodeFront = []
        for n in nodeFront:
          if not n in visitedNodes:
            visitedNodes.add(n)
            if self.debugLevel > 0:
              print("      DEBUG: colorNodesLevelAuto visiting %s" %
                    n.getName())
            for e in n.getFanoutMainFlowEdges():
              if edge2color.get(e, None) == None:
                edge2color[e] = color
                toNode = e.getToNode()
                nodeSup = toNode.isSupported()
                if  nodeSup == sgSupported:
                  # Consider for expansion
                  if all(edge2color.get(eIn, None) == color for eIn in toNode.getFaninMainFlowEdges()):
                    self.setNodeColor(toNode, color)
                    newNodeFront.append(toNode)
                    change = True
                    if self.debugLevel > 0:
                      print("    DEBUG: colorNodesLevelAuto EXPANDED setColor %d on %-12s %s" %
                            (self.getNodeColor(toNode), toNode.getOpType(), toNode.getName()))
                else:
                  newNodeFrontNext.append(toNode)
        nodeFront = newNodeFront
        if self.debugLevel > 0:
          print("    DEBUG: colorNodesLevelAuto new wavefront ",
                [n.getName() for n in nodeFront])
      nodeFront = newNodeFrontNext
      color = self.getNewColor()
      sgSupported = not sgSupported
      if self.debugLevel > 0:
        print("  DEBUG: colorNodesLevelAuto starting new color %d wavefront " % color,
              [n.getName() for n in nodeFront])
  

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

  # Color nodes to define subgraph partitions - start new partition from a multi node
  # The from list defines multi-nodes (cut levels) as comma-separated nodes
  def colorNodesFromMulti(self, fromMultiNodeList):
    sourceGraph = self.__kgraph
    node2cut = {}  # example:  node2cut[a] = "a,b"
    cut2color = {}
    for cut in fromMultiNodeList:
      cutNodes = cut.split(',')
      for n in cutNodes:
        node2cut[n] = cut
      cut2color[cut] = None
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
        print("DEBUG: colorNodesFromMulti visit         %-12s %s" %
              (n.getOpType(), n.getName()))
      # Coloring
      if n.getName() in node2cut:
        cut = node2cut[n.getName()]
        if cut2color[cut] == None:
          color = self.getNewColor()
          cut2color[cut] = color
        else:
          color = cut2color[cut]
        self.setNodeColor(n, color)
      else:
        self.setNodeColor(n, color)
      fanoutEdges = n.getFanoutMainFlowEdges()
      edgeQueue += [(e, color) for e in fanoutEdges]
      if self.debugLevel > 0:
        print("DEBUG: colorNodesFromMulti setColor %d on %-12s %s" %
              (self.getNodeColor(n), n.getOpType(), n.getName()))

  # Overrides existing colors using Node Color .Node1 Color1 ... list
  def adjustColor(self, nodeColorAdjustment):
    for nodeName,color in zip(nodeColorAdjustment[::2], [int(x) for x in nodeColorAdjustment[1::2]]):
      node = self.__kgraph.getNode(nodeName)
      oldColor = self.getNodeColor(node)
      if not oldColor == color:
        self.setNodeColor(node, color)
        if color + 1 > self.__numColors:
          self.__numColors = color + 1
        print("INFO: overrode color %d to %d on node %s" % (oldColor, color, node.getName()))

  class Cut(object):
    def __init__(self, nodes, visitedNodes, srcKgraph, costFunctionName, debugLevel):
      self.nodes = nodes
      self.visitedNodes = visitedNodes # visitedNodes maintains full single-color subgraph created by cut expansion
      #assert type(visitedNodes) == 'set'
      self.srcKgraph = srcKgraph
      self.debugLevel = debugLevel
      self.costFunctionName = costFunctionName

    def getCost(self):
      cost = 0
      allNodes = self.visitedNodes | set(self.nodes)
      for n in allNodes:
        costFunction = getattr(n, self.costFunctionName)
        cost += costFunction(n)
        #if self.debugLevel > 0:
        #  print("      DEBUG: colorNodesMultiTpb  %-12s  %s has  %d  ops" % (n.getOpType(), n.getName(), costFunction(n)))
      return cost

    # Greedy search for a next cut with the cost closest to the target
    def expand(self, costTarget, allVisited):
      cuts = []
      if self.debugLevel > 0:
        print("DEBUG: colorNodesMultiTpb expanding nodes %s " % str([n.getName() for n in self.nodes]))
      if len(self.nodes) == 0:
        return self
      for n in self.nodes:
        nextNodes = n.getFanoutMainFlowNodes()
        newNodes = self.nodes.copy()
        fullyExapandedN = True
        for nextNode in nextNodes:
          if nextNode not in (set(newNodes) | self.visitedNodes | allVisited):
            if nextNode.hasMainFlowPredecessorsInSet(self.visitedNodes | set([n])):
              newNodes.append(nextNode)
            else:
              fullyExapandedN = False
        newVisited = self.visitedNodes.copy()
        if fullyExapandedN:
          newNodes.remove(n)
        if not n in allVisited:
          newVisited.add(n)
        if not set(self.nodes) == set(newNodes):
          cuts.append(KgraphPart.Cut(newNodes, newVisited, self.srcKgraph, self.costFunctionName, self.debugLevel))
          if self.debugLevel > 0:
            print("    DEBUG: colorNodesMultiTpb expanded ### %s #### ---->  $$$ %s $$$" %
                  (str([n.getName() for n in self.nodes]), str([n.getName() for n in newNodes])))
      if len(cuts) == 0:
        # We should try to expand from all subsets of nodes. For simplicity start with all nodes (sufficient for resnet/googlenet like)
        newNodes = []
        for n in self.nodes:
          for nextNode in n.getFanoutMainFlowNodes():
            if nextNode not in (set(newNodes) | self.visitedNodes | allVisited):
              assert nextNode.hasMainFlowPredecessorsInSet(self.visitedNodes | set(self.nodes))
              newNodes.append(nextNode)
        newVisited = self.visitedNodes.copy()
        for n in self.nodes:
          if not n in allVisited:
            newVisited.add(n)
        cuts.append(KgraphPart.Cut(newNodes, newVisited, self.srcKgraph, self.costFunctionName, self.debugLevel))
      bestCuts = sorted(cuts, key=lambda x : abs(x.getCost() - costTarget))
      if len(bestCuts) > 0:
        if self.debugLevel > 0:
          print("  DEBUG: colorNodesMultiTpb cost [Gop] is %.3f of %.3f" % (bestCuts[0].getCost() / 1e9,  costTarget/1e9))
        return bestCuts[0]
      else:
        return KgraphPart.Cut([], self.visitedNodes | set(self.nodes), self.srcKgraph, self.costFunctionName, self.debugLevel)

  # Color nodes to a pipeline of TPBs of about equal Op count
  def colorNodesMultiTpb(self, numTpbs):
    sourceGraph = self.__kgraph
    
    # Op counts
    srcTotNumOps = 0
    for n in sourceGraph.getNodes():
      srcTotNumOps += n.getOpCount()
    cutOpTarget = 1.0 * srcTotNumOps / numTpbs
    
    cut = KgraphPart.Cut(sourceGraph.getInputNodes(), set(), sourceGraph, 'getOpCount', self.debugLevel)
    allVisited = set()
    while len(cut.nodes) > 0:
      color = self.getNewColor()
      while len(cut.nodes) > 0 and cut.getCost() < cutOpTarget:
        cut = cut.expand(cutOpTarget, allVisited)
        allVisited |= cut.visitedNodes
      for n in cut.visitedNodes:
        assert self.getNodeColor(n) == None
        self.setNodeColor(n, color)
        if self.debugLevel > 0:
          print("DEBUG: colorNodesMultiTpb colored %d  %-12s %s " %
                (color, n.getOpType(), n.getName()))
      cut = KgraphPart.Cut(cut.nodes, set(), sourceGraph, 'getOpCount', self.debugLevel)


  # Color nodes given the partitioning strategy
  # The strategy is a keyword and arguments (for some)
  def colorNodes(self, partitioningStrategy, nodeColorAdjustment):
    strategy = partitioningStrategy[0]
    if strategy == "auto":
      self.colorNodesAuto()
    elif strategy == "meauto":
      self.colorNodesMeAuto()
    elif strategy == "conv":
      self.colorNodesConv()
    elif strategy == "suppauto":
      self.colorNodesSuppAuto()
    elif strategy == "levelauto":
      self.colorNodesLevelAuto()
    elif strategy == "from":
      self.colorNodesFrom( partitioningStrategy[1:])
    elif strategy == "from_multi":
      self.colorNodesFromMulti( partitioningStrategy[1:])
    elif strategy == "multi_tpb":
      numTpbs = float(partitioningStrategy[1])
      self.colorNodesMultiTpb(numTpbs)
    else:
      assert 0
    if len(nodeColorAdjustment) > 0:
      self.adjustColor(nodeColorAdjustment)

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
          if self.nodeIsSgOutput(n):
            subGraph.addOutput(nCopy)
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
  
  # Report SG properties such as ops, weights, if and of map sizes
  def reportOpsAndSizes(self):
    propNp = np.zeros((self.__numColors, 4))
    for i in range(self.__numColors):
      sg = self.__subgraphs[i]
      opCount, weightSize, ifmapSize, ofmapSize = sg.graph.getOpsAndSizes(sg.getInputs(), sg.getOutputs())
      propNp[i] = [opCount / 1e9, weightSize / 2**20, ifmapSize / 2**20, ofmapSize / 2**20]
    # Statistics
    print("%-4s    %-7s  %-7s  %-7s  %-7s" % ("SG", 'opCountGop', 'weightSizeMiB', 'ifmapSizeMiB', 'ofmapSizeMiB'))
    for i in range(self.__numColors):
      print("sg%02d  " %i, "%7.3f  %7.3f  %7.3f  %7.3f" % (tuple(propNp[i])))
    print("%-36s" % ("OpsG mean std  min max"))
    ops = propNp[...,0]
    print("%12.2f %12.2f  %12.2f %12.2f" % (ops.mean(), ops.std(), ops.min(), ops.max()))

  
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

def attachPrePost(sgJsonList, preprocessor, postprocessor, preprocessor_args, postprocessor_args):
  for (sgname) in ["sg_pre", "sg_post"]:
    if sgname == "sg_pre":
      f = preprocessor
      cmd_args = preprocessor_args
    else:
      f = postprocessor
      cmd_args = postprocessor_args
    if f != "":
      sgDir = sgname
      print("\nINFO: processing subgraph %s" % sgDir)
      os.makedirs(sgname)
      print("INFO: cmd : " + f)
      assert(os.path.isfile(f) and os.access(f, os.X_OK))
      shutil.copy2(f, os.getcwd() + "/" + sgname)
      sgJson = {}
      sgJson["executor"] = "processor"
      sgJson["SubGraphDir"] = sgname
      sgJson["cmd"] =  os.path.basename(f)
      sgJson["cmd_args"] = cmd_args
      sgJson["Inputs"] = []
      sgJson["Outputs"] = []
      if sgname == "sg_pre":
        sgJson["Outputs"] += sgJsonList[0]["Inputs"]
        sgJsonList.insert(0, sgJson)
      else:
        sgJson["Inputs"] += sgJsonList[-1]["Outputs"]
        sgJsonList.append(sgJson)











