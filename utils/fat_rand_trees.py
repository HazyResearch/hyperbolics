import networkx as nx
import numpy as np

def gen_fat_random_tree(n, d, dpth, p):
    G = nx.balanced_tree(d, dpth)
    m = G.order()

    par_idx = 0
    for i in range(dpth): par_idx += d**i

    i = m
    while i in range(m , n):
        if np.random.uniform() < p:
            G.add_edge(par_idx, i)
            #print("(", par_idx, " , ", i, ")")
            i += 1

        par_idx += 1
        if par_idx > m: par_idx = 1+d

    #print(G.order(), "\n", G.edges())
    return G

n = 50
d = 4
dpth = 2
p = 0.3

num_trees = 2
for i in range(num_trees):
    G = gen_fat_random_tree(n, d, dpth, p)
    nx.write_edgelist(G, str(i) + "_small_tree" + ".edges", data=False)
