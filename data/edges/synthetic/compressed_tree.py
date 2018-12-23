import numpy as np
import networkx as nx
import sys, os
import subprocess

# generate some random trees on the same nodes:
n = 300
t = 5
g_list = []

for i in range(t):
    g_list.append(nx.random_tree(n))
    
# compress the tree:
G = nx.Graph()
for node in range(n):
    for tree in range(t):
        for edge in g_list[tree].edges(node):
            G.add_edge(edge[0], edge[1])

nx.write_edgelist(G, 'compressed_tree.edges', data=False)
