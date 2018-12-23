import numpy as np
import networkx as nx
import itertools
import argh


# construct generalized Sierpinski graph
# vertices: strings of length d chosen from [n]

def construct(n=3, d=2, base='clique'):
    if base in ['clique', 'K', 'k']:
        base = 'K'
        base_graph = list(nx.complete_graph(n).edges)
    if base in ['cycle', 'C', 'c']:
        base = 'C'
        base_graph = list(nx.cycle_graph(n).edges)

    G = nx.Graph()
    for t in range(0, d):
        choices = [list(range(n))]*t
        for prefix in itertools.product(*choices):
            # for p in range(n):
            #     for q in range(p):
            for p,q in base_graph:
                    a = list(prefix) + [p] + [q]*(d-t-1)
                    b = list(prefix) + [q] + [p]*(d-t-1)
                    G.add_edge(tuple(a), tuple(b))


    def idx(L, base):
        if len(L) == 1: return L[0]
        return L[-1] + base*idx(L[:-1], base)

    mapping = {L : idx(list(L), n) for L in itertools.product(*([list(range(n))]*d))}
    G = nx.relabel_nodes(G, mapping, copy=False)

    nx.write_edgelist(G, f"sierp-{base}{n}-{d}.edges", data=False)

if __name__ == '__main__':
    _parser = argh.ArghParser()
    _parser.set_default_command(construct)
    _parser.dispatch()
