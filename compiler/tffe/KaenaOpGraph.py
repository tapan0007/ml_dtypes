# Kaena absration of neural network framework operations
# Suitable for TF freeze graph input and general op reduction and fusing


class Object:
  def __init__(self, name, attrs):
    self.__name = name
    self.__attrs = attrs
  def getName(self):
    return(self.__name)
  def getAttr(self, attrName):
    return(self.__attrs[attrName])
  def setAttr(self, attrName, attrVal):
    self.__attrs[attrName] = attrVal

class Node(Object):
  def __init__(self, name, attrs):
    Object.__init__(self, name, attrs)
    self.__level = -1
  def __str__(self):
    return("Node=" + self.getName())
  def getLevel(self):
    return(self.__level)
  def setLevel(self, level):
    self.__level = level
  def getNpInfo(self):
    return(self.getName(),
           self.getAttr("op_type"),
           self.getAttr("np_shape"),
           self.getAttr("np_file"))
           

class Edge(Object):
  def __init__(self, fromNode, toNode, attrs):
    Object.__init__(self, "Edge", attrs)
    self.__fromNode = fromNode
    self.__toNode = toNode
  def fromNode(self):
    return(self.__fromNode)
  def toNode(self):
    return(self.__toNode)
  def __str__(self):
    return("Edge  From=" + self.fromNode().getName()
           + "  To=" + self.toNode().getName())

class Graph(Object):
  def __init__(self, name, attrs = {}):
    Object.__init__(self, name, attrs)
    self.__name2node = {}
    self.__edges = {}
    
  def addNode(self, name, attrs = {}):
    self.__name2node[name] = Node(name, attrs)
  
  def addEdge(self, nameFrom, nameTo, attrs = {}):
    fromNode = self.getNode(nameFrom)
    toNode = self.getNode(nameTo)
    if self.__edges.get(fromNode) == None:
      self.__edges[fromNode] = {}
    self.__edges[fromNode][toNode] = Edge(fromNode, toNode, attrs)

  def hasNode(self, name):
    return(self.__name2node.get(name, None))

  def getNode(self, name):
    return(self.__name2node[name])
  
  def getEdge(self, nameFrom, NameTo):
    return(self.__edges[nameFrom][nameTo])
  
  def getNodes(self):
    return(list(self.__name2node.values()))
  
  def getEdges(self):
    edges = []
    for fromNode in self.__edges:
      edges += list(self.__edges[fromNode].values())
    return(edges)
  
  def nodeSuccessors(self, fromNode):
    nextNodes = []
    if self.__edges.get(fromNode):
      nextNodes = list(self.__edges[fromNode].keys())
    return(nextNodes)
  
  def nodePredecessors(self, toNode):
    preNodes = []
    for fromNode in self.__edges.keys():
      if toNode in self.__edges[fromNode]:
        preNodes.append(fromNode)
    return(preNodes)
  
  def getTopNode(self):
    nextNodes = self.getNodes()
    while len(nextNodes) > 0:
      n = nextNodes[0]
      nextNodes = self.nodeSuccessors(n)
    return(n)
        
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
  



