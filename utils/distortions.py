# distortions.py
# python code to compute distortion/MAP
import numpy as np
import scipy.sparse.csgraph as csg
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx

def entry_is_good(h, h_rec): return (not np.isnan(h_rec)) and (not np.isinf(h_rec)) and h_rec != 0 and h != 0

def distortion_entry(h,h_rec,me,mc):
    avg = abs(h_rec - h)/h
    if h_rec/h > me: me = h_rec/h
    if h/h_rec > mc: mc = h/h_rec
    return (avg,me,mc)

def distortion_row(H1, H2, n, row):
    mc, me, avg, good = 0,0,0,0
    for i in range(n):
        if i != row and entry_is_good(H1[i], H2[i]):
            (_avg,me,mc) = distortion_entry(H1[i], H2[i],me,mc)
            good        += 1
            avg         += _avg
    avg /= good if good > 0 else 1.0
    return (mc, me, avg, n-1-good)

def distortion(H1, H2, n, jobs):
    H1, H2 = np.array(H1), np.array(H2)
    dists = Parallel(n_jobs=jobs)(delayed(distortion_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    dists = np.vstack(dists)
    mc = max(dists[:,0])
    me = max(dists[:,1])
    # wc = max(dists[:,0])*max(dists[:,1])
    avg = sum(dists[:,2])/n
    bad = sum(dists[:,3])
    return (mc, me, avg, bad)

def map_via_edges(G, i, h_rec):
    neighbors   = set(map(int, G.getrow(i).indices))
    sorted_dist = np.argsort(h_rec)
    m           = len(neighbors)
    precs       = np.zeros(m)
    n_correct   = 0
    j = 0
    n = h_rec.size
    n_idx = np.array(list(neighbors), dtype=np.int)
    sds   = sorted_dist[1:(m+1)]
    # print(f"{n_idx} {type(n_idx)} {n_idx.dtype}")
    # print(f"i={i} neighbors={neighbors} {sds} {h_rec[n_idx]} {h_rec[sds]}")
    # skip yourself, you're always the nearest guy
    for i in range(1,n):
        if sorted_dist[i] in neighbors:
            n_correct += 1
            precs[j] = n_correct/float(i)
            j += 1
            if j == m:
                break
    return np.sum(precs)/min(n,m)
    # return np.sum(precs)/j


def map_row(H1, H2, n, row, verbose=False):
    edge_mask = (H1 == 1.0)
    m         = np.sum(edge_mask).astype(int)
    assert m > 0
    if verbose: print(f"\t There are {m} edges for {row} of {n}")
    d = H2
    sorted_dist = np.argsort(d)
    if verbose:
        print(f"\t {sorted_dist[0:5]} vs. {np.array(range(n))[edge_mask]}")
        print(f"\t {d[sorted_dist[0:5]]} vs. {H1[edge_mask]}")
    precs       = np.zeros(m)
    n_correct   = 0
    j = 0
    # skip yourself, you're always the nearest guy
    # TODO (A): j is redundant here
    for i in range(1,n):
        if edge_mask[sorted_dist[i]]:
            n_correct += 1
            precs[j] = n_correct/float(i)
            j += 1
            if j == m:
                break
    return np.sum(precs)/m

def map_score(H1, H2, n, jobs):
    #maps = Parallel(n_jobs=jobs)(delayed(map_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    maps  = [map_row(H1[i,:],H2[i,:],n,i) for i in range(n)]
    return np.sum(maps)/n
