import numpy as np
import networkx as nx
import itertools
import argh

cycle_nodes = 10

tree = nx.balanced_tree(2, 2)
nx.relabel_nodes(tree, {n : n+1 for n in tree.nodes}, copy=False)
tree.add_edge(0, 1)
tree_nodes = len(tree.nodes())

copies = []
for i in range(cycle_nodes):
    T = tree.copy()
    copies.append(nx.relabel_nodes(T, {n : cycle_nodes * n + i for n in T.nodes}))
G = nx.compose_all(copies + [nx.cycle_graph(cycle_nodes)])
# G = nx.compose_all(copies)

nx.write_edgelist(G, "cycle-tree.edges", data=False)

