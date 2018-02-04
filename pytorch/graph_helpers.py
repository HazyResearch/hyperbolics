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
    nChunks     = 128 if num_worers > 1 else n
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
