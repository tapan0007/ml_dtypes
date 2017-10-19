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

class Node(Object):
  def __str__(self):
    return("Node=" + self.getName())

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
    return(self.__name2node.get(name) != None)

  def getNode(self, name):
    return(self.__name2node[name])
  
  def getEdge(self, nameFrom, NameTo):
    return(self.__name2node[name])
  
  def getNodes(self):
    return(self.__name2node.values())
  
  def getEdges(self):
    edges = []
    for fromNode in self.__edges:
      edges += list(self.__edges[fromNode].values())
    return(edges)



