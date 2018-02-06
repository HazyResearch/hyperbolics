# Should be moved to utility
from multiprocessing import Pool
import networkx as nx
import scipy.sparse.csgraph as csg
import logging
import numpy as np

def djikstra_wrapper( _x ):
    (mat, x) = _x
    return csg.dijkstra(mat, indices=x, unweighted=True, directed=False)

def build_distance(G, scale, num_workers=None):
    n = G.order()
    p = Pool() if num_workers is None else Pool(num_workers)
    
    adj_mat_original = nx.to_scipy_sparse_matrix(G)

    # Simple chunking
    nChunks     = 128 if num_workers is not None and num_workers > 1 else n
    if n > nChunks:
        chunk_size  = n//nChunks
        extra_chunk_size = (n - (n//nChunks)*nChunks)
        logging.info(f"\tCreating {nChunks} of size {chunk_size} and an extra chunk of size {extra_chunk_size}")

        chunks     = [ list(range(k*chunk_size, (k+1)*chunk_size)) for k in range(nChunks)]
        if extra_chunk_size >0: chunks.append(list(range(n-extra_chunk_size, n)))
        Hs = p.map(djikstra_wrapper, [(adj_mat_original, chunk) for chunk in chunks])
        H  = np.concatenate(Hs,0)
        logging.info(f"\tFinal Matrix {H.shape}")
    else:
        H = djikstra_wrapper( (adj_mat_original, list(range(n))) )
        
    H *= scale
    return H

def build_distance_hyperbolic(G, scale):
    return np.cosh(build_distance(G,scale)-1.)/2.

def dist_sample_rebuild(dm, alpha):
    dist_mat = np.copy(dm)
    n,_ = dist_mat.shape    
  
    keep_dists = np.random.binomial(1,alpha,(n,n))
    
    # sample matrix:
    for i in range(n):
        for j in range(n):
            dist_mat[i,j] = -1 if keep_dists[i,j] == 0 and i!=j else dist_mat[i,j]
       
    # use symmetry first for recovery:
    for i in range(n):
        for j in range(i+1,n):
            if dist_mat[i,j] == -1 and dist_mat[j,i] > 0:
                dist_mat[i,j] = dist_mat[j,i]
            if dist_mat[j,i] == -1 and dist_mat[i,j] > 0:
                dist_mat[j,i] = dist_mat[i,j]
                
    # now let's rebuild it with triangle inequality:
    largest_dist = np.max(dist_mat)
    
    for i in range(n):
        for j in range(i+1,n):
            # missing distance:
            if dist_mat[i,j] == -1:
                dist = largest_dist

                for k in range(n):
                    if dist_mat[i,k] > 0 and dist_mat[j,k] > 0 and dist_mat[i,k]+dist_mat[j,k] < dist:
                        dist = dist_mat[i,k]+dist_mat[j,k]

                    dist_mat[i,j] = dist
                    dist_mat[j,i] = dist
    return dist_mat