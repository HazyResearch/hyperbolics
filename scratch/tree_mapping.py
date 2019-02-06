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
import mapping_utils as util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

euclidean_embeddings = {}
saved_tensors = os.listdir("tree_emb_saved/")
indices = []
for file in saved_tensors:
    idx = int(file.split(".")[0])
    indices.append(idx)
    euclidean_embeddings[idx] = torch.load("tree_emb_saved/"+str(file), map_location=torch.device('cpu'))


#Riemannian SGD

from torch.optim.optimizer import Optimizer, required
spten_t = torch.sparse.FloatTensor


def poincare_grad(p, d_p):
    """
    Calculates Riemannian grad from Euclidean grad.
    Args:
        p (Tensor): Current point in the ball
        d_p (Tensor): Euclidean gradient at p
    """
    if d_p.is_sparse:
        p_sqnorm = torch.sum(
            p.data[d_p._indices()[0].squeeze()] ** 2, dim=1,
            keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)

    return d_p



def euclidean_grad(p, d_p):
    return d_p


def retraction(p, d_p, lr):
    # Gradient clipping.
    if torch.all(d_p < 1000) and torch.all(d_p>-1000):
        p.data.add_(-lr, d_p)


class RiemannianSGD(Optimizer):
    r"""Riemannian stochastic gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rgrad (Function): Function to compute the Riemannian gradient from
            an Euclidean gradient
        retraction (Function): Function to update the parameters via a
            retraction of the Riemannian gradient
        lr (float): learning rate
    """

    def __init__(self, params, lr=required, rgrad=required, retraction=required):
        defaults = dict(lr=lr, rgrad=rgrad, retraction=retraction)
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """Performs a single optimization step.
        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group['lr']
                d_p = group['rgrad'](p, d_p)
                group['retraction'](p, d_p, lr)

        return loss


# Does Euclidean to hyperbolic mapping using series of FC layers.
# We use ground truth distance matrix for the pair since the distortion for hyperbolic embs are really low.

def trainFCHyp(input_matrix, ground_truth, n, mapping, mapping_optimizer):
    mapping_optimizer.zero_grad()
 
    loss = 0
    output = mapping(input_matrix.float())
    dist_recovered = util.distance_matrix_hyperbolic(output) 
    loss += util.distortion(ground_truth, dist_recovered, n)
    loss.backward()
    mapping_optimizer.step()

    return loss.item()


def trainFCIters(mapping, n_epochs=5, n_iters=500, print_every=50, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  
    plot_loss_total = 0  

    mapping_optimizer = RiemannianSGD(mapping.parameters(), lr=learning_rate, rgrad=poincare_grad, retraction=retraction)
    training_pairs = [util.pairfromidx(idx) for idx in range(n_iters)]

    for i in range(n_epochs):
        print("Starting epoch "+str(i))
        iter=1
        for idx in indices:     
            input_matrix = euclidean_embeddings[idx]
            target_matrix = training_pairs[idx][1]
            n = training_pairs[idx][2]
            loss = trainFCHyp(input_matrix, target_matrix, n, mapping, mapping_optimizer)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (util.timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            iter+=1
            
input_size = 10
output_size = 10
mapping = nn.Sequential(
          nn.Linear(input_size, 50).to(device),
          nn.ReLU().to(device),
          nn.Linear(50, output_size).to(device),
          nn.ReLU().to(device))
          
trainFCIters(mapping)




