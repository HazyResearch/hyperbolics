# load_dist.py
import networkx as nx
import numpy as np
import pickle
import scipy.sparse.csgraph as csg
from joblib import Parallel, delayed
import multiprocessing
import data_prep as dp
import time

def compute_row(i, adj_mat): 
    return csg.dijkstra(adj_mat, indices=[i], unweighted=True, directed=False)
    
def save_dist_mat(G, file):
    n = G.order()
    print("Number of nodes is ", n)
    adj_mat = nx.to_scipy_sparse_matrix(G)
    t = time.time()
    
    num_cores = multiprocessing.cpu_count()
    dist_mat = Parallel(n_jobs=20)(delayed(compute_row)(i,adj_mat) for i in range(n))
    dist_mat = np.vstack(dist_mat)
    print("Time elapsed = ", time.time()-t)
    pickle.dump(dist_mat, open(file,"wb"))

def load_dist_mat(file):
    return pickle.load(open(file,"rb"))

def get_dist_mat(G):
    n = G.order()
    print("Number of nodes is ", n)
    adj_mat = nx.to_scipy_sparse_matrix(G)
    t = time.time()
    
    num_cores = multiprocessing.cpu_count()
    dist_mat = Parallel(n_jobs=20)(delayed(compute_row)(i,adj_mat) for i in range(n))
    dist_mat = np.vstack(dist_mat)
    print("Time elapsed = ", time.time()-t)
    return dist_mat