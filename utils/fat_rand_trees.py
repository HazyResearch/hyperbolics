import networkx as nx
import numpy as np

# generates a random tree that is "fat", complete near root, random below
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

# generates a tree that's close (within a number of edges) to a given graph
def nearby_tree(G, delta_edges):
    G2 = G.copy()

    leaves = []
    for (node, deg) in G2.degree():
        if deg == 1:
            leaves.append(node)

    prm = np.random.permutation(leaves)

    # TODO: careful if there's not enough leaves
    for i in range(delta_edges):
        leaf    = prm[i]
        neighbor = list(G2.edges(leaf))[0][1]
        G2.remove_edge(leaf, neighbor)
        
        print("(leaf, neighbor) = ", leaf, neighbor)

        cands = [x for x in list(G.nodes()) if x not in (leaf, neighbor)]
        cands = np.random.permutation(cands)
        G2.add_edge(leaf, cands[0])

        print("new (leaf, neighbor) = ", leaf, cands[0])
    
    return G2

# n = 50
# d = 4
# dpth = 2
# p = 0.3

def make_trees(num_trees, n, d, dpth, p, folder, name, do_nearby, delta_edges, nearby_folder):
    for i in range(num_trees):
        G = gen_fat_random_tree(n, d, dpth, p)
        nx.write_edgelist(G, folder + "/" + str(i) + name + ".edges", data=False)

        if do_nearby:
            G2 = nearby_tree(G, delta_edges)
            nx.write_edgelist(G2, nearby_folder + "/" + str(i) + name + ".nearby.edges", data=False)