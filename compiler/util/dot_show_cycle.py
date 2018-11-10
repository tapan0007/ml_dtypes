import networkx as nx
G = nx.drawing.nx_pydot.read_dot("trivnet_graph.dot")

cycles = nx.simple_cycles(G)
for c in list(cycles):
  print("\n", c)

