# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Neural network scheduler
#

import networkx as nx
import graphviz as gv
import re
import nnsh_cost
import nnsh_ops

class NnshGraph(object):

  t2c = {
    "Conv"     : "red",
    "MaxPool"  : "orange",
    "AvgPool"  : "orange",
    "BiasAdd"  : "blue",
    "ResAdd"   : "cyan",
    "Const"    : "green",
    }
  def type2color(self, layerType):
    return NnshGraph.t2c.get(layerType, "black")

  def __init__(self, debug, jsonData=None):
    self.nxg = nx.DiGraph()
    self.debug = debug
    self.numFusedNodes = 0
    if not jsonData == None:
      self.dType = jsonData["data_type"]
      for layer in jsonData["layers"]:
        # node
        attrs = {}
        if not "data_type" in layer:
          layer["data_type"] = self.dType
        nodeName = layer["layer_name"]
        layerType = layer["layer_type"]
        shape = layer["ofmap_shape"]
        dotText = "%s\n%s\n%s" % (layerType, nodeName, shape)
        if True:
          dotText = "\n".join(["%s : %s" % (k, v) for k,v in sorted(layer.items())]) 
        #attrs["color"] = self.type2color(layerType)
        attrs["shape"] = "rect"
        kAttrs = layer
        self.nxg.add_node(nodeName, shape="rect", label=dotText,
                         layer_type=layerType, k_attrs=kAttrs)
        self.nxg.node[nodeName]["color"] = self.type2color(layerType)
        #print(dot.source)
        # edge
        for prevNode in layer["previous_layers"]:
          edgeAttrs = {}
          self.nxg.add_edge(prevNode, nodeName)
          #print(dot.source)
  
  def copy(self):
    new = NnshGraph(self.debug)
    new.nxg = self.nxg.copy()
    return new
  
  def render(self, outFile):
    outFileNx = outFile
    nx.nx_pydot.write_dot(self.nxg, outFileNx)
    with open(outFileNx, "r") as fh:
      dotText = fh.read()
    dot = gv.Source(dotText)
    dot.format = "svg"
    dot.render(outFileNx)
    print("INFO: wrote  %s.%s" % (outFileNx, "svg"))
    
  def getSource(self):
    #sources = [node for node, degree in self.nxg.in_degree() if degree == 0]
    sources = [node for node, data in self.nxg.nodes(data=True) if data["layer_type"] == "Input"]
    assert len(sources) == 1
    return sources[0]
  
  def getLayerType(self, node):
    return self.nxg.node[node]["layer_type"]
  
  def pathMatchesReList(self, path, ruleNodeRe):
    if self.debug > 2:
      print("DEBUG: testing path %s agains %s" % (path, ruleNodeRe))
    if len(path) == len(ruleNodeRe):
      for pos in range(len(ruleNodeRe)):
        n = path[pos]
        layerType = self.getLayerType(n)
        #print("DEBUG   comparing pos %d  %s  %s" % (pos, layerType, ruleNodeRe[pos]))
        if not re.match(ruleNodeRe[pos], layerType):
          return False
      return True
    else:
      return False
  
  def fusePathToNode(self, ruleName, path):
    newNode = "%s_%02d" % (ruleName, self.numFusedNodes)
    self.numFusedNodes += 1
    newLabel = "layer_name : %s\nlayer_type : %s\n\n" % (newNode, ruleName)
    for n in path:
      newLabel += self.nxg.node[n]["label"] + "\n\n"
    self.nxg.add_node(newNode, layer_type=ruleName, shape="rect",
                      sub_graph=path, label=newLabel)
    for pred in self.nxg.predecessors(path[0]):
      self.nxg.add_edge(pred, newNode)
    for succ in self.nxg.successors(path[-1]):
      self.nxg.add_edge(newNode, succ)
    self.nxg.remove_nodes_from(path)
  
  def fuse(self, fuseRules):
    srcNode = self.getSource()
    visitedNodes = set()
    nodesToFuse = []
    for ruleName, ruleNodeRe in fuseRules.items():
      for n in nx.topological_sort(self.nxg):
        if not n in visitedNodes:
          for p in nx.single_source_shortest_path(self.nxg, n, len(ruleNodeRe)).values():
            if self.pathMatchesReList(p, ruleNodeRe):
              if self.debug > 0:
                print("DEBUG: matched %s" % p)
              visitedNodes.union(p)
              nodesToFuse.append([ruleName, p])
    for ruleName, path in nodesToFuse:
      self.fusePathToNode(ruleName, path)

  def fold(self, foldOpSet):
    srcNode = self.getSource()
    nodesToFold = []
    for n in nx.topological_sort(self.nxg):
      for nPred in self.nxg.predecessors(n):
        if self.getLayerType(nPred) in foldOpSet:
          self.nxg.node[n]["k_const"] = self.nxg.node[nPred]["k_attrs"]
          nodesToFold.append(nPred)
          if self.debug > 0:
            print("DEBUG: folded Const %s" % nPred)
    self.nxg.remove_nodes_from(nodesToFold)

  def selectNextOp(self, stateNodes, node2succ):
    stateNodes = [n for n in stateNodes if len(node2succ[n]) > 0]
    for n in stateNodes:
      if self.debug > 1:
        print("  DEBUG: testing state node %s" % (n))
      for ns in node2succ[n]:
        if self.debug > 1:
          print("    DEBUG: testing next node %s" % (ns))
        if all(np in stateNodes for np in self.nxg.predecessors(ns)):
          if self.debug > 1:
            print("      DEBUG: selected next node %s" % (ns))
          for nt in node2succ:  # loop needed for multi-fanin like resadd
            if ns in node2succ[nt]:
              node2succ[nt].remove(ns)
          return ns, stateNodes, node2succ
    return None, stateNodes, node2succ

  def getCuts(self):
    #bfsEdges = nx.bfs_edges(self.nxg, self.getSource())
    #bfsEdgesList = list(bfsEdges)
    #print("DEBUG: profile edges %s" % bfsEdgesList)
    
    src = self.getSource()
    stateNodes = [src]
    cuts = [stateNodes.copy()]
    node2succ = {src : list(self.nxg.successors(src))}
    while 1:
      n, stateNodes, node2succ = self.selectNextOp(stateNodes, node2succ)
      if n == None:
        break
      cut = [n] + stateNodes
      cuts.append(cut)        
      if self.debug > 0:
        print("  DEBUG: cut  ", cut)
      node2succ[n] = list(self.nxg.successors(n))
      stateNodes.append(n)
    if self.debug > 1:
      print("DEBUG: cuts  ", cuts)
    return cuts
  
  def createOpfromNode(self, node):
    layerType = self.nxg.node[node]["layer_type"]
    layerAttrs = self.nxg.node[node]
    cl = eval("nnsh_ops.Nnsh" + layerType)
    op = cl(node, layerAttrs)
    return(op)
    
  # The frirst node in the cut is the op being calculated
  # The rest is ifmaps (state)
  def getCutCosts(self, cutNodes):
    opCosts = []
    for i in range(len(cutNodes)):
      n = cutNodes[i]
      op = self.createOpfromNode(n)
      if i == 0:
        c = op.getOpCost()
      else:
        c = op.getStateCost()
      opCosts.append(c)
    return opCosts








