import networkx as nx
import random
import argparse
import datetime

def getInfoDateStr():
  return "INFO %s :" % str(datetime.datetime.now())

class Frontier:
  def __init__(self, graph, nodes=[]):
    self.graph = graph
    self.nodes = list(set(nodes))
  def expandRandom(self):
    if len(self.nodes) > 0:
      idx = random.randint(0, len(self.nodes)-1)
      n = self.nodes[idx]
      #print('DEBUG: selected node', n)
      return Frontier(self.graph, set(self.nodes + list(self.graph.successors(n))) - set([n]))
    else:
      return Frontier(self.graph)
  def expandMin(self):
    if len(self.nodes) > 0:
      frontiers = []
      for n in self.nodes:
        f = Frontier(self.graph, set(
          self.nodes + list(self.graph.successors(n))) - set([n]))
        frontiers.append(f)
      minLen = len(frontiers[0].nodes)
      for f in frontiers:
        if len(f.nodes) < minLen:
          minLen = len(f.nodes)
      minFrontiers = [f for f in frontiers if len(f.nodes) <= minLen]
      idx = random.randint(0, len(minFrontiers)-1)
      f = minFrontiers[idx]
      #print('DEBUG: selected min frontier', f.nodes, 'out of', len(frontiers))
      return f
    else:
      return Frontier(self.graph)
  def getNodes(self):
    return list(set(self.nodes))
  def __eq__(self, other):
    return self.getNodes() == other.getNodes()

class LevelCuts:
  def __init__(self, dotFile, userInputs):
    print(getInfoDateStr(), 'loading ', dotFile)
    self.graph = nx.drawing.nx_pydot.read_dot(dotFile)
    g = self.graph
    print(getInfoDateStr(), 'loaded graph with %d nodes %d edges' % (len(g), g.size()))
    self.outputs = [n for n in g.nodes if g.out_degree(n) == 0]
    # User overrides for inputs
    if userInputs[0] == 'indegree0':
      self.inputs = [n for n in g.nodes if g.in_degree(n) == 0]
    elif userInputs[0] == 'type':
      typeStr = userInputs[1]
      self.inputs = [n for n in g.nodes if typeStr in self.graph.nodes[n]['label']]
    else:
      self.inputs = [n for n in userInputs if n in self.graph.nodes]
    print(getInfoDateStr(), 'identified %d input(s)\n' % len(self.inputs),
            "  ".join(self.inputs))
  
  def getCuts(self, step, seed, strategy):
    prevFrontier = Frontier(self.graph)
    frontier = Frontier(self.graph, self.inputs)
    cuts = []
    random.seed(seed)
    while frontier != prevFrontier:
      for i in range(step):
        prevFrontier = frontier
        if strategy == 'random':
          frontier = prevFrontier.expandRandom()
        elif strategy == 'min':
          frontier = prevFrontier.expandMin()
        else:
          raise ValueError('ERROR: unknown strategy ' + strategy)
      cuts.append(frontier.getNodes())
    return cuts

parser = argparse.ArgumentParser()
parser.add_argument('--dot', help='Graph input file', default="trivnet_graph.dot")
parser.add_argument('--inputs', help='Input selection: indegree0 or type Placeholder (the default) or name1 name2 ...', nargs='+', default=['type', 'Placeholder'])
parser.add_argument('--step', help='How many expansiosn to perform between cuts, larger number yields smaller number of cuts', type=int, default=1)
parser.add_argument('--seed', help='Seed for the random number of generator to select diffent cuts, default is 19', type=int, default=19)
parser.add_argument('--strategy', help='Cut strategy, min (default), random', default='min')
args = parser.parse_args()

lc = LevelCuts(args.dot, args.inputs)

cuts = lc.getCuts(args.step, args.seed, args.strategy)
for c in cuts:
  print(','.join(c))
print(getInfoDateStr(), 'indentified %d cuts' % (len(cuts)))


