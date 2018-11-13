import torch
import numpy as np
import os, sys
from sklearn.manifold import Isomap
import utils.distortions as dis
import utils.load_graph as load_graph

module_path = os.path.abspath(os.path.join('./pytorch'))
if module_path not in sys.path:
    sys.path.append(module_path)

import graph_helpers as gh
from hyperbolic_models import ProductEmbedding
from hyperbolic_parameter import RParameter  

def unwrap(x):
    if isinstance(x, list) : return [unwrap(u) for u in x]
    if isinstance(x, tuple): return tuple([unwrap(u) for u in list(x)])
    return x.detach().cpu().numpy()

def dist_e(u, v):
    return np.linalg.norm(u-v)

def dist_row(x, i):
    m = x.shape[0]
    dx = np.zeros([m])

    for j in range(m):
        dx[j] = dist_e(x[i,:], x[j,:])
    
    return dx

def dist_matrix(x):
    m = x.shape[0]
    rets = np.zeros([m,m])
    for i in range(m):
        rets[i,:] = dist_row(x, i)
        #print(rets)
    return rets

# load an embedding and a graph and do isomap
def run_isomap(emb_name, dataset, r):
    #emb_name = 'isomap_test/smalltree.E10-1.lr10.emb.final'
    #dataset = 'data/edges/smalltree.edges'
    dataset = 'data/edges/' + dataset + '.edges'
    m = torch.load(emb_name)

    emb_orig = unwrap(m.E[0].w)

    # perform the isomap dim reduction
    embedding = Isomap(n_components=r)
    emb_transformed = embedding.fit_transform(emb_orig)

    #print(emb_transformed.shape)

    num_workers = 1
    scale = 1

    # compute d_avg
    G = load_graph.load_graph(dataset)
    n = G.order()
    H = gh.build_distance(G, scale, num_workers=int(num_workers) if num_workers is not None else 16)

    #Hrec = unwrap(m.dist_matrix())
    Hrec = dist_matrix(emb_transformed)
    mc, me, avg_dist, nan_elements = dis.distortion(H, Hrec, n, num_workers)
    wc_dist = me*mc

    print("d_avg = ", avg_dist)
    return avg_dist