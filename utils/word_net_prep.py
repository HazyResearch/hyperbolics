from nltk.corpus import wordnet as wn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import floyd_warshall, connected_components
from collections import defaultdict
import numpy as np
import networkx as nx

# for adding edges in CSR format
def make_edge_set(): return ([],([],[]))
def add_edge(e, i,j):
    (v,(row,col)) = e
    row.append(i)
    col.append(j)
    v.append(1)


def load_wordnet():
    d        = dict()
    all_syns = list(wn.all_synsets('n'))
    for idx, x in enumerate(all_syns): d[x] = idx
    n         = len(all_syns)
    e = make_edge_set()
    for idx, x in enumerate(all_syns):
        for y in x.hypernyms():
            y_idx = d[y]
            add_edge(e, idx  , y_idx)
            # add_edge(e, y_idx,   idx)
    return csr_matrix(e,shape=(n, n))

def load_big_component():
    X = load_wordnet()
    C = connected_components(X, directed=False)
    sizes = [0] * C[0]
    for i in C[1]:
        sizes[i] += 1
    # sizes = np.array(sizes)
    big_comp_idx = np.argmax(sizes)
    # print(f"{C[0]} connected components")
    # print("connected components sizes: ", sizes[sizes > 1])
    print(f"big_comp_idx ", big_comp_idx)

    all_syns = list(wn.all_synsets('n'))
    comp_0 = np.array(all_syns)[C[1] == big_comp_idx]
    n_f    = len(comp_0)
    _d     = dict()
    for idx, x in enumerate(comp_0): _d[x] = idx

    # e_f = make_edge_set()
    # closure = make_edge_set()
    e_f = []
    closure = []
    for idx, x in enumerate(comp_0):
        for y in x.hypernyms():
            y_idx = _d[y]
            # add_edge(e_f, idx  , y_idx)
            # add_edge(e_f, y_idx,   idx)
            e_f.append((y_idx, idx))
        for y in x.closure(lambda z: z.hypernyms()):
            y_idx = _d[y]
            # add_edge(closure, idx, y_idx)
            closure.append((idx, y_idx))

    # X2 = csr_matrix(e_f, shape=(n_f,n_f))
    G = nx.DiGraph(e_f)
    G_closure = nx.DiGraph(closure)
    return (n_f, G, G_closure)

if __name__ == '__main__':
    n, G, G_closure = load_big_component()
    # edges = nx.from_scipy_sparse_matrix(G)
    nx.write_edgelist(G, f"data/edges/wordnet2.edges", data=False)
    nx.write_edgelist(G_closure, f"data/edges/wordnet_closure.edges", data=False)
