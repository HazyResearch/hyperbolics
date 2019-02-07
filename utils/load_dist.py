# load_dist.py
import networkx as nx
import numpy as np
import pickle
import scipy.sparse.csgraph as csg
from joblib import Parallel, delayed
import multiprocessing
#import data_prep as dp
import time
import torch
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0, parentdir+"/pytorch")
import pytorch.hyperbolic_models

def compute_row(i, adj_mat): 
    return csg.dijkstra(adj_mat, indices=[i], unweighted=True, directed=False)
    
def save_dist_mat(G, file):
    n = G.order()
    print("Number of nodes is ", n)
    adj_mat = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))
    t = time.time()
    
    num_cores = multiprocessing.cpu_count()
    dist_mat = Parallel(n_jobs=20)(delayed(compute_row)(i,adj_mat) for i in range(n))
    dist_mat = np.vstack(dist_mat)
    print("Time elapsed = ", time.time()-t)
    pickle.dump(dist_mat, open(file,"wb"))

def load_dist_mat(file):
    return pickle.load(open(file,"rb"))

def unwrap(x):
    """ Extract the numbers from (sequences of) pytorch tensors """
    if isinstance(x, list) : return [unwrap(u) for u in x]
    if isinstance(x, tuple): return tuple([unwrap(u) for u in list(x)])
    return x.detach().cpu().numpy()

def load_emb_dm(file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = torch.load(file).to(device)
    H = unwrap(m.dist_matrix())
    return H

def get_dist_mat(G, parallelize=True):
    n = G.order()
    print("Number of nodes is ", n)
    adj_mat = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))
    t = time.time()

    num_cores = multiprocessing.cpu_count() if parallelize else 1

    dist_mat = Parallel(n_jobs=num_cores)(delayed(compute_row)(i,adj_mat) for i in range(n))
    dist_mat = np.vstack(dist_mat)
    print("Time elapsed = ", time.time()-t)
    return dist_mat
