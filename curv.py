import logging, argh
import os, sys
import networkx as nx
import numpy as np

# root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, root_dir)
import utils.load_graph as load_graph
import utils.vis as vis
import utils.distortions as dis
import pytorch.graph_helpers as gh


def Ka(D, m, b, c, a):
    if a == m: return 0.0
    k = D[a][m]**2 + D[b][c]**2/4.0 - (D[a][b]**2 + D[a][c]**2)/2.0
    k /= D[a][m]
    print(f'{m}; {b} {c}; {a}: {k}')
    return k

def K(D, n, m, b, c):
    ks = [Ka(D, m, b, c, a) for a in range(n)]
    return np.mean(ks)


def estimate_curvature(G, D, n):
    for m in range(n):
        ks = []
        edges = list(G.edges(m))
        for i in range(len(edges)):
            for j in range(b,len(edges)):
                b = edges[i]
                c = edges[j]
                ks.append(K(D, n, b, c))
        # TODO turn ks into a cdf

    return None

# TODO: what is the correct normalization wrt n? e.g. make sure it works for d-ary trees
def sample_K(G, D, n, n_samples=100):
    samples = []
    _cnt = 0;
    while _cnt < n_samples:
        m = np.random.randint(0, n)
        edges = list(G.edges(m))
        print(f"edges of {m}: {edges}")
        i = np.random.randint(0, len(edges))
        j = np.random.randint(0, len(edges))
        b = edges[i][1]
        c = edges[j][1]
        # TODO special case for b=c?
        if b==c: continue
        a = np.random.randint(0, n)
        k = Ka(D, m, b, c, a)
        samples.append(k)
        print(k)

        _cnt += 1

    return samples


# @argh.arg('--dataset')
def estimate(dataset='data/edges/smalltree.edges'):
    G = load_graph.load_graph(dataset)
    n = G.order()
    GM = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))

    num_workers = 16
    D   = gh.build_distance(G, 1.0, num_workers) # load the whole matrix

    samples = sample_K(G, D, n, 100)
    print(np.mean(samples), np.std(samples)**2)


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_default_command([estimate])
    parser.dispatch()
