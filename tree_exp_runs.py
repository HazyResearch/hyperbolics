from __future__ import unicode_literals, print_function, division
import os
import subprocess
import itertools
import numpy as np
import argh
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
import utils.mapping_utils as util
import pdb
import glob


def run_hmds(run_name):
    ranks = [10]
    print(os.listdir(f"{run_name}/emb/"))
    for file in os.listdir(f"{run_name}/emb/"):
        print(file)
        file_base = file.split('.')[0]
        cmd_base  = "julia hMDS/hmds-simple.jl"
        cmd_edges = " -d data/edges/random_tree_edges/" + file_base + ".edges"
        cmd_emb   = f" -k {run_name}/emb/{file}"
        cmd_rank  = " -r "
        cmd_scale = " -t "
        for rank in ranks:
            print("Rank = ", rank)
            best_distortion=1e5
            best_mapval=0.0
            best_edge_acc =0.0
            best_scale=1.0
            for i in range(10):
                scale = 0.1*(i+1)
                cmd = cmd_base + cmd_edges + cmd_emb + cmd_rank + str(rank) + cmd_scale + str(scale)
                result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
                res_string = result.stdout.decode('utf-8')
                res_lines = res_string.splitlines()

                #grab values from hMDS results
                curr_distortion = np.float64(res_lines[16].split()[5].strip(","))
                curr_mapval     = np.float64(res_lines[17].split()[2])
                curr_edge_acc   = np.float64(res_lines[21].split()[7])

                if curr_distortion < best_distortion:
                    best_distortion = curr_distortion
                    best_scale = scale
                if curr_mapval > best_mapval:
                    best_mapval = curr_mapval
                if curr_edge_acc > best_edge_acc:
                    best_edge_acc = curr_edge_acc

                if i == 0:
                    input_distortion = res_lines[18].split()[6].strip(",")
                    input_map        = res_lines[19].split()[3]
                    input_edge_acc   = res_lines[20].split()[5]
                    print("Input distortion \t", input_distortion, "\t input mAP \t", input_map, "\t input Edge Acc from MST \t", input_edge_acc, "\n")

            print("Best scale \t", str(best_scale), "\t Best distortion \t", str(best_distortion), "\t Best mAP \t", str(best_mapval), "\t Best edge Acc from MST \t", str(best_edge_acc)) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.optim import Optimizer
from torch.optim.optimizer import Optimizer, required

def run_net(run_name, edge_folder):   
    print(device)
    euclidean_embeddings = {}
    saved_tensors = os.listdir(f"{run_name}/emb/")
    indices = []

    for file in saved_tensors:
        idx = int(file.split(".")[0])
        indices.append(idx)
        embedding = torch.load(f"{run_name}/emb/"+str(file), map_location=torch.device('cpu'))
        norm = embedding.norm(p=2, dim=1, keepdim=True)
        max_norm = torch.max(norm)+1e-10
        normalized_emb = embedding.div(max_norm.expand_as(embedding))
        euclidean_embeddings[idx] = normalized_emb

    def poincare_grad(p, d_p):
        """
        Calculates Riemannian grad from Euclidean grad.
        Args:
            p (Tensor): Current point in the ball
            d_p (Tensor): Euclidean gradient at p
        """
        p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
        return d_p

    def _correct(x, eps=1e-10):
            current_norms = torch.norm(x,2,x.dim() - 1)
            mask_idx      = current_norms < 1./(1+eps)
            modified      = 1./((1+eps)*current_norms)
            modified[mask_idx] = 1.0
            return modified.unsqueeze(-1)


    def euclidean_grad(p, d_p):
        return d_p

    def retraction(p, d_p, lr):
        # Gradient clipping.
        d_p.clamp_(min=-10000, max=10000)
        p.data.add_(-lr, d_p)
        #project back to the manifold.
        p.data = p.data * _correct(p.data)

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

    def trainFCHyp(input_matrix, target_matrix, ground_truth,  n, mapping, mapping_optimizer):
        mapping_optimizer.zero_grad()
        loss = 0
        output = mapping(input_matrix.float())
        dist_recovered = util.distance_matrix_hyperbolic(output) 
        loss += util.distortion(target_matrix, dist_recovered, n)
        dummy = dist_recovered.clone()
        edge_acc = util.compare_mst(ground_truth, dummy.detach().numpy())
        loss.backward(retain_graph=True)
        mapping_optimizer.step()
        return loss.item(), edge_acc


    def trainFCIters(mapping, n_epochs=30, n_iters=200, print_every=10, plot_every=100, learning_rate=1.0):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  
        plot_loss_total = 0  
        

        mapping_optimizer = RiemannianSGD(mapping.parameters(), lr=learning_rate, rgrad=poincare_grad, retraction=retraction)
        training_pairs = [util.pairfromidx(edge_folder, idx) for idx in range(len(indices))]

        for idx in range(n_epochs):
            print("Starting epoch "+str(idx))
            iter=1
            if idx!=0 and idx%2==0:
                mapping_optimizer = RiemannianSGD(mapping.parameters(), lr=learning_rate*0.5, rgrad=poincare_grad, retraction=retraction)
            for i in indices:
                input_matrix = euclidean_embeddings[i]
                target_matrix = training_pairs[i][1]
                n = training_pairs[i][2]
                G = training_pairs[i][3]
                loss, edge_acc = trainFCHyp(input_matrix, target_matrix, G, n, mapping, mapping_optimizer)
                print_loss_total += loss
                plot_loss_total += loss


                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (util.timeSince(start, iter / n_iters),
                                                iter, iter / n_iters * 100, print_loss_avg))
                    print("Edge accuracy", edge_acc)

                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                iter+=1
                
    input_size = 10
    output_size = 10
    mapping = nn.Sequential(
            nn.Linear(input_size, 1000).to(device),
            nn.ReLU().to(device),
            nn.Linear(1000, 100).to(device),
            nn.ReLU().to(device),
            nn.Linear(100, output_size).to(device),
            nn.ReLU().to(device))
    
            
    return trainFCIters(mapping)


@argh.arg("run_name", help="Directory to store the run")

def run(run_name, edge_folder="./data/edges/random_tree_edges/"):
    print("Running hMDS")
    run_hmds(run_name)
    print("Running Net")
    run_net(run_name, edge_folder)

if __name__ == '__main__':
    _parser = argh.ArghParser()
    _parser.set_default_command(run)
    _parser.dispatch()
