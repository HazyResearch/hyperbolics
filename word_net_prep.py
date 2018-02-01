from nltk.corpus import wordnet as wn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import floyd_warshall, connected_components
from collections import defaultdict
import numpy as np

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
            add_edge(e, y_idx,   idx)
    return csr_matrix(e,shape=(n, n))

def load_big_component():
    X = load_wordnet()
    C = connected_components(X)
    z = defaultdict(int)
    for i in C[1]: z[i] += 1
    all_syns = list(wn.all_synsets('n'))
    comp_0 = np.array(all_syns)[C[1] == 0]
    n_f    = len(comp_0)
    _d     = dict()
    for idx, x in enumerate(comp_0): _d[x] = idx

    e_f = make_edge_set()
    for idx, x in enumerate(comp_0):
        for y in x.hypernyms():
            y_idx = _d[y]
            
            add_edge(e_f, idx  , y_idx)
            add_edge(e_f, y_idx,   idx)
        
    X2  = csr_matrix(e_f, shape=(n_f,n_f))
    return (n_f, X2)

