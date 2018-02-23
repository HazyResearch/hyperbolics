# This implements the algorithm for finding a good tree embedding from 
import networkx as nx
import scipy.sparse.csgraph as csg
import numpy as np
import time, argh
import data_prep as dp
import distortions as dis
import load_dist as ld
import pickle

from joblib import Parallel, delayed
import multiprocessing


# get the biggest Gromov product in num rows
def biggest_row(metric,start,num, r, n):
    p,q,curr = 0,0,-1
    for a in range(start, min(start+num, n)-1):
        if metric[a] >= 0:
            for b in range(n):
                if metric[b] >= 0 and a != b:
                    gpr = gp(dists,a,b,r)
                    if gpr >= curr:
                        p,q,curr = (a,b,gpr)    
    return (p,q,curr)        

# get a node from G
def first_node(G):
    for node in G.nodes():
        return node

# helper to run Dijkstra
def compute_row(i, adj_mat, uw): 
    return csg.dijkstra(adj_mat, indices=[i], unweighted=uw, directed=False)
		
# the Gromov product distance
def gp(dists,x,y,z):
    dxy = dists[x,y]
    dxz = dists[x,z]
    dyz = dists[y,z]

    return 1/2*(float(dxz+dyz-dxy))

# iterative construction. Requires that nodes are 0...n-1
def construct_tree_i(metric, r, next_label, n):  
  
    # build list of nodes to remove and re-attach, until we get down to 1 node...
    removal, partner =  np.zeros(n-2), np.zeros(n-2);
    removal*=-1;
        
    metric_shared = np.array(metric)
    metric_shared[r] = -1    
 
    with Parallel(n_jobs=-1) as parallel:
        idx = 0
        while sum(metric_shared>=0)>1:
            t = time.time()
            num=100
            res = parallel(delayed(biggest_row)(metric_shared, idx*num, num, r, n) for idx in range(int(np.ceil(n/num))))      
            
            biggest_rows = np.vstack(res)
            biggest = np.argmax(biggest_rows[:,2])
            p = int(biggest_rows[biggest,0])
            q = int(biggest_rows[biggest,1])
            if dists[p,r] > dists[q,r]:
                p,q = q,p
        
            removal[idx] = q
            partner[idx] = p
            metric_shared[q] = -1
            idx += 1
            #print("Elapsed = ", time.time()-t)

    # put in the first node:
    v = np.argmax(metric_shared)
    T = nx.Graph()
    T.add_edge(int(v),int(r),weight=dists[v,r])
    idx -= 1

    # place the remaining nodes one by one:
    while idx >= 0:
        q,p = int(removal[idx]), int(partner[idx])
        qr_p = gp(dists,q,r,p)
        pr_q = gp(dists,p,r,q)
        pq_r = gp(dists,p,q,r)

        # get the new weight for the Steiner node and add it in
        for node in T.neighbors(p):
            new_weight = max(0,T[p][node]["weight"]-qr_p)
            T.add_edge(next_label, node, weight=new_weight)

        # reattach p and q as leaves
        T.remove_node(p)
        T.add_edge(next_label, p, weight=qr_p)        
        T.add_edge(q, next_label, weight=pr_q)
        next_label += 1
        idx -= 1

    return T


@argh.arg("--ds", help="Dataset")

def steiner_tree(ds="1"):
    ds = int(ds)
    G = dp.load_graph(ds)
    n = G.order()
    print("# of vertices is ", n)

    global dists 
    dists = ld.load_dist_mat("dists/dist_mat"+str(ds)+".p")    

    nv = np.zeros(n)
    for i in range(n):
        nv[i] = 1

    metric = list(G.nodes())
    
    # root:
    r = first_node(G)

    print("Building trees")

    t = time.time()
    G_tree = construct_tree_i(metric, r, n, n) 
    print("Done. Elapsed time = ", time.time()-t)
    
    n_Steiner = G_tree.order()

    adj_mat_tree = nx.to_scipy_sparse_matrix(G_tree.to_undirected(), range(n_Steiner))    
    dist_mat_S_tree = Parallel(n_jobs=20)(delayed(compute_row)(i,adj_mat_tree, False) for i in range(n_Steiner))
    dist_mat_S_tree = np.vstack(dist_mat_S_tree)	
    dist_mat_S_tree_n = dist_mat_S_tree[0:n, 0:n]
    
    print("Measuring Distortion")
    t = time.time()
    t_d_max, t_d_avg, bad = dis.distortion(dists, dist_mat_S_tree_n, n, 2)
    print("Steiner tree distortion = ", t_d_max, t_d_avg)

    MAP_S = dis.map_score(dists, dist_mat_S_tree_n, n, 2)
    print("Steiner tree MAP = ", MAP_S)
    
    file = "./trees/tree" + str(ds) + ".p"
    pickle.dump(G_tree, open(file,"wb"))    

    print("Elapsed time = ", time.time()-t)
    return G_tree

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    _parser.add_commands([steiner_tree])
    _parser.dispatch()
