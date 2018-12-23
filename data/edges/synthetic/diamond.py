import numpy as np
import networkx as nx
import itertools
import argh

d = 6

edges = [(0,1), (1,2), (2,3), (3,0)]
n = 4
for t in range(d-1):
    edges2 = []
    for u,v in edges:
        edges2 += [(u, n), (n, v), (v, n+1), (n+1, u)]
        n += 2
    edges = edges2


nx.write_edgelist(nx.Graph(edges), f"diamond{d}.edges", data=False)
