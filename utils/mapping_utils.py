from __future__ import unicode_literals, print_function, division
import os
import numpy as np
import scipy
import scipy.sparse.csgraph as csg
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
from io import open
import unicodedata
import string
import re
import random
import json
from collections import defaultdict
import utils.load_dist as ld

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Distortion calculations

def acosh(x):
    return torch.log(x + torch.sqrt(x**2-1))

def _correct(x, eps=1e-1):
        current_norms = torch.norm(x,2,x.dim() - 1)
        mask_idx      = current_norms < 1./(1+eps)
        modified      = 1./((1+eps)*current_norms)
        modified[mask_idx] = 1.0
        return modified.unsqueeze(-1)

def dist_h(u,v):
    u = u * _correct(u)
    v = v * _correct(v)
    z  = 2*torch.norm(u-v,2)**2
    uu = 1. + torch.div(z,((1-torch.norm(u,2)**2)*(1-torch.norm(v,2)**2)))
    return acosh(uu)

def dist_p(u,v):
    z  = 2*torch.norm(u-v,2)**2
    uu = 1. + torch.div(z,((1-torch.norm(u,2)**2)*(1-torch.norm(v,2)**2)))
    machine_eps = np.finfo(uu.data.detach().cpu().numpy().dtype).eps  # problem with cuda tensor
    return acosh(torch.clamp(uu, min=1+machine_eps))
    #return acosh(uu)


def distance_matrix_euclidean(input):
    row_n = input.shape[0]
    mp1 = torch.stack([input]*row_n)
    mp2 = torch.stack([input]*row_n).transpose(0,1)
    dist_mat = torch.sum((mp1-mp2)**2,2).squeeze()
    return dist_mat

def distance_matrix_hyperbolic(input, sampled_rows):
    #print("were computing the matrix with sampled_rows = ")
    #print(sampled_rows)
    row_n = input.shape[0]
    dist_mat = torch.zeros(len(sampled_rows), row_n, device=device)
    # num_cores = multiprocessing.cpu_count()
    # dist_mat = Parallel(n_jobs=num_cores)(delayed(compute_row)(i,adj_mat) for i in range(n))
    idx = 0
    for row in sampled_rows:
        for i in range(row_n):
            if i != row:
                dist_mat[idx, i] = dist_p(input[row,:], input[i,:])
        idx += 1
    # print("Distance matrix", dist_mat)
    return dist_mat

def entry_is_good(h, h_rec): return (not torch.isnan(h_rec)) and (not torch.isinf(h_rec)) and h_rec != 0 and h != 0

def distortion_entry(h,h_rec):
    avg = abs(h_rec - h)/h
    return avg

def distortion_row(H1, H2, n, row):
    avg, good = 0, 0
    for i in range(n):
        if i != row and entry_is_good(H1[i], H2[i]):
            _avg = distortion_entry(H1[i], H2[i])
            good        += 1
            avg         += _avg
    if good > 0:
        avg /= good 
    else:
        avg, good = torch.tensor(0., device=device, requires_grad=True), torch.tensor(0., device=device, requires_grad=True)
    # print("Number of good entries", good)
    return (avg, good)

def distortion(H1, H2, n, sampled_rows, jobs=16):
    # dists = Parallel(n_jobs=jobs)(delayed(distortion_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    i = 0
    dists = torch.zeros(len(sampled_rows))
    #print(dists)
    for row in sampled_rows:
        #print(H1[row,:])
        #print(H2[i,:])
        #print(n)
        #print(row)
        #print("i = ", i)
        dists[i] = distortion_row(H1[row,:], H2[i,:], n, row)[0]
        i += 1

    #to_stack = [tup[0] for tup in dists]
    #avg = torch.stack(to_stack).sum() / len(sampled_rows)
    avg = dists.sum() / len(sampled_rows)
    return avg
'''

def distortion(H1, H2, n, jobs):
    H1 = np.array(H1.cpu()), 
    H2 = np.array(H2.detach().cpu())
    dists = Parallel(n_jobs=jobs)(delayed(distortion_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    dists = np.vstack(dists)
    mc = max(dists[:,0])
    me = max(dists[:,1])
    # wc = max(dists[:,0])*max(dists[:,1])
    avg = sum(dists[:,2])/n
    bad = sum(dists[:,3])
    #return (mc, me, avg, bad)    
    to_stack = [tup[0] for tup in dists]
    avg = torch.stack(to_stack).sum()/n
    return avg
'''


#Loading the graph and getting the distance matrix.

def load_graph(file_name, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    with open(file_name, "r") as f:
        for line in f:
            tokens = line.split()
            u = int(tokens[0])
            v = int(tokens[1])
            if len(tokens) > 2:
                w = float(tokens[2])
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(u,v)
    return G


def compute_row(i, adj_mat): 
    return csg.dijkstra(adj_mat, indices=[i], unweighted=True, directed=False)

def get_dist_mat(G):
    n = G.order()
    adj_mat = nx.to_scipy_sparse_matrix(G, nodelist=list(range(G.order())))
    t = time.time()
    
    num_cores = multiprocessing.cpu_count()
    dist_mat = Parallel(n_jobs=num_cores)(delayed(compute_row)(i,adj_mat) for i in range(n))
    dist_mat = np.vstack(dist_mat)
    return dist_mat

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def pairfromidx(idx, edge_folder):
    G = load_graph(edge_folder+str(idx)+".edges")
    target_matrix = get_dist_mat(G)
    target_tensor = torch.from_numpy(target_matrix).float().to(device)
    target_tensor.requires_grad = False
    n = G.order()
    return ([], target_tensor, n, G)

def gettestpairs(test_folder):
    test_pairs = defaultdict()
    edge_files = os.listdir(test_folder+"edges/")
    for file in edge_files:
        name = file.split("/")[-1]
        ground_truth = load_graph(test_folder+"edges/"+file)
        n = ground_truth.order()
        euclidean_emb = torch.load(test_folder+"emb_tensor/"+str(name)+".E10-1.lr10.0.emb.final", map_location=torch.device('cpu'))
        target_matrix = get_dist_mat(ground_truth)
        target_tensor = torch.from_numpy(target_matrix).float().to(device)
        target_tensor.requires_grad = False
        test_pairs[name] = [euclidean_emb, ground_truth, target_tensor, n]
    return test_pairs

def compare_mst(G, hrec):

    mst = csg.minimum_spanning_tree(hrec)
    G_rec = nx.from_scipy_sparse_matrix(mst)
    #np.set_printoptions(threshold=np.inf)
    #print(hrec)

    #print(G.edges())
    #print("ours")
    #print(G_rec.edges())

    found = 0
    for edge in G_rec.edges():
        if edge in G.edges(): found+= 1

    acc = found / len(list(G.edges()))
    return acc
