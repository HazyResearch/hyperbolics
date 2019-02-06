from __future__ import unicode_literals, print_function, division
import os
import numpy as np
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Distortion calculations

def acosh(x):
    return torch.log(x + torch.sqrt(x**2-1))

def dist_h(u,v):
    z  = 2*torch.norm(u-v,2)**2
    uu = 1. + torch.div(z,((1-torch.norm(u,2)**2)*(1-torch.norm(v,2)**2)))
    return acosh(uu)

def distance_matrix_euclidean(input):
    row_n = input.shape[0]
    mp1 = torch.stack([input]*row_n)
    mp2 = torch.stack([input]*row_n).transpose(0,1)
    dist_mat = torch.sum((mp1-mp2)**2,2).squeeze()
    return dist_mat

def distance_matrix_hyperbolic(input):
    row_n = input.shape[0]
    dist_mat = torch.zeros(row_n, row_n, device=device)
    for row in range(row_n):
        for i in range(row_n):
            if i != row:
                dist_mat[row, i] = dist_h(input[row,:], input[i,:])
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
    return (avg, good)

def distortion(H1, H2, n, jobs=16):
#     dists = Parallel(n_jobs=jobs)(delayed(distortion_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    dists = (distortion_row(H1[i,:],H2[i,:],n,i) for i in range(n))
    to_stack = [tup[0] for tup in dists]
    avg = torch.stack(to_stack).sum()/n
    return avg


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


def pairfromidx(idx):
    G = load_graph("random_trees_edges/"+str(idx)+".edges")
    target_matrix = get_dist_mat(G)
    target_tensor = torch.from_numpy(target_matrix).float().to(device)
    target_tensor.requires_grad = False
    n = G.order()
    return ([], target_tensor, n, [])