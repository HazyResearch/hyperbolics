# distortions.py
# python code to compute distortion/MAP
import numpy as np
import scipy.sparse.csgraph as csg
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx

def distortion_row(H1, H2, n, row):
    mc, me, avg, good = 0,0,0,0
    for i in range(n):
        if i != row:
            if (not np.isnan(H2[i])) and (not np.isinf(H2[i])) and H2[i] != 0 and H1[i] != 0:
                #avg += max(H2[i]/H1[i], H1[i]/H2[i]);
                avg += abs(H2[i]-H1[i])/H1[i];
                                
                if H2[i]/H1[i] > me:
                    me = H2[i]/H1[i]

                if H1[i]/H2[i] > mc:
                    mc = H1[i]/H2[i]

                good += 1
    avg /= (good);
    return (mc, me, avg, n-1-good)

def distortion(H1, H2, n, jobs):
    H1, H2 = np.array(H1), np.array(H2)
    dists = Parallel(n_jobs=jobs)(delayed(distortion_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    dists = np.vstack(dists)
    wc = max(dists[:,0])*max(dists[:,1])
    avg = sum(dists[:,2])/n
    bad = sum(dists[:,3])
    return (wc, avg, bad)

def map_row(H1, H2, n, row):
    edge_mask = (H1 == 1.0).astype(float)
    m = sum(edge_mask).astype(int)
    d = H2
    sorted_dist = np.argsort(d)
    precs = np.zeros(m)    
    
    n_correct = 0
    j = 0
    # skip yourself, you're always the nearest guy    
    for i in range(1,n): 
        n_correct += 1
        precs[j] = n_correct/float(i)
        j += 1
        if j == m:
            break
            
    return sum(precs)/m 

def map_score(H1, H2, n, jobs):
    maps = Parallel(n_jobs=jobs)(delayed(map_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    return sum(maps)/n 

        
# test    
#H1 = np.array([[0,1,1],[1,0,1],[1,1,0]])
#H2 = np.array([[0,4,6],[4,0,10],[6,10,0]])
