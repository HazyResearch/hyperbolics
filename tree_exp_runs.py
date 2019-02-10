from __future__ import unicode_literals, print_function, division
import os
import subprocess
import logging
import itertools
from collections import defaultdict
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
import utils.distortions as dis
#import utils.learning_util as lu
import warnings
from torch.optim import Optimizer
from torch.optim.optimizer import Optimizer, required

def run_hmds(run_name):
    ranks = [8]
    logging.info(f"Starting hMDS experiment:")
    print(os.listdir(f"{run_name}/hmds_emb/"))
    for file in os.listdir(f"{run_name}/hmds_emb/"):
        print(file)
        logging.info(f"Working with the embedding: {file}")
        file_base = file.split('.')[0]
        cmd_base  = "julia hMDS/hmds-simple.jl"
        cmd_edges = f" -d {run_name}/test/edges/{file_base}.edges"
        cmd_emb   = f" -k {run_name}/hmds_emb/{file}"
        cmd_rank  = " -r "
        cmd_scale = " -t "
        for rank in ranks:
            print("Rank = ", rank)
            best_distortion=1e5
            best_mapval=0.0
            best_edge_acc =0.0
            best_scale=1.0
            for i in range(1):
                scale = 0.1*(i+1)
                cmd = cmd_base + cmd_edges + cmd_emb + cmd_rank + str(rank) + cmd_scale + str(scale)
                #print(cmd)
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


            input_distortion = res_lines[18].split()[6].strip(",")
            input_map        = res_lines[19].split()[3]
            input_edge_acc   = res_lines[20].split()[5]
            print(f"input distortion {input_distortion}")
            logging.info(f"input distortion {input_distortion}")
            print("input map", input_map)
            logging.info(f"input map {input_map}")
            print("input Edge Acc from MST", input_edge_acc)
            logging.info(f"input Edge Acc from MST {input_edge_acc}")

            print("Best scale \t", str(best_scale), "\t Best distortion \t", str(best_distortion), "\t Best mAP \t", str(best_mapval), "\t Best edge Acc from MST \t", str(best_edge_acc)) 
            logging.info(f"For rank {rank}:")
            logging.info(f"Best scale: {best_scale}") 
            logging.info(f"Best distortion: {best_distortion}"), 
            logging.info(f"Best mAP: {best_mapval}")
            logging.info(f"Best Edge Acc from MST: {best_edge_acc}")

            input_distortion, input_map, input_edge_acc = np.float64(input_distortion), np.float64(input_map), np.float64(input_edge_acc)
            with open(run_name+"/"+str(file)+"."+str(rank)+'rank.stat', "w") as f:
                f.write("InDist InMAP InEdgeAcc Best-Scale Best-dist Best-MAP Best-EdgeAcc \n")
                f.write(f"{input_distortion:10.6f} {input_map:8.4f} {input_edge_acc:8.4f} {best_scale:8.4f} {best_distortion:10.6f} {best_mapval:8.4f} {best_edge_acc:8.4f}")


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

def run_net(run_name, edge_folder, test_folder, device):
    logging.info("Starting net:")

    euclidean_embeddings = {}
    saved_tensors = os.listdir(f"{run_name}/emb/")
    indices = []

    for file in saved_tensors:
        idx = int(file.split(".")[0])
        indices.append(idx)
        #md = torch.load(f"{run_name}/emb/"+str(file), map_location=torch.device('cpu'))
        md = torch.load(f"{run_name}/emb/"+str(file), map_location=device)
        embedding = md.E[0].w
        norm = embedding.norm(p=2, dim=1, keepdim=True)
        max_norm = torch.max(norm)+1e-10
        normalized_emb = embedding.div(max_norm.expand_as(embedding))
        euclidean_embeddings[idx] = normalized_emb

    input_size = 10
    output_size = 9
    mapping = nn.Sequential(
        nn.Linear(input_size, 1000).to(device),
        nn.ReLU().to(device),
        nn.Linear(1000, 100).to(device),
        nn.ReLU().to(device),
        nn.Linear(100, output_size).to(device),
        nn.ReLU().to(device))

    return trainFCIters(mapping, indices, edge_folder, euclidean_embeddings)


# Does Euclidean to hyperbolic mapping using series of FC layers.
def trainFCHyp(input_matrix, target_matrix, ground_truth, n, mapping, sampled_rows, mapping_optimizer, verbose=False):
    st = time.time()
    mapping_optimizer.zero_grad()
    loss = 0
    if verbose: print("Zeroed gradients, ", time.time()-st)
    st = time.time()

    output = mapping(input_matrix.float())
    if verbose: print("Did the mapping, ", time.time()-st)
    st = time.time()

    dist_recovered = util.distance_matrix_hyperbolic(output, sampled_rows) 
    if verbose: print("Found the distance, ", time.time()-st)
    st = time.time()

    loss += util.distortion(target_matrix, dist_recovered, n, sampled_rows, 50)
    if verbose: print("Computed the loss, ", time.time()-st)
    st = time.time()

    #dummy = dist_recovered.clone()
    #edge_acc = util.compare_mst(ground_truth, dummy.detach().cpu().numpy())
    #print("Got edge accs, ", time.time()-st)
    #st = time.time()

    loss.backward(retain_graph=True)
    if verbose: print("Backprop, ", time.time()-st)
    st = time.time()

    mapping_optimizer.step()
    if verbose: print("Took step, ", time.time()-st)
    st = time.time()
    
    #return loss.item(), edge_acc
    return loss.item(), 0


def full_stats_pass_point(input_matrix, target_matrix, ground_truth, n, mapping, verbose=False):
    st = time.time()
    output = mapping(input_matrix.float())

    # look at every row
    sampled_rows = list(range(input_matrix.shape[0]))
    
    dist_recovered = util.distance_matrix_hyperbolic(output, sampled_rows) 
    dis = util.distortion(target_matrix, dist_recovered, n, sampled_rows, 50).detach().cpu().numpy()
    if verbose: print("Computed the distortion for this matrix, ", time.time()-st)
    st = time.time()

    dummy = dist_recovered.clone()
    edge_acc = util.compare_mst(ground_truth, dummy.detach().cpu().numpy())
    if verbose: print("Got edge accs, ", time.time()-st)

    return dis, edge_acc

def trainFCIters(mapping, indices, edge_folder, euclidean_embeddings, n_epochs=10000, print_every=5, learning_rate=0.2):
    start = time.time()
    print_loss_total = 0
    plot_loss_total = 0
    n_iters = len(indices)
    subsample_row_num = 10
    resample_every = 5
    full_stats_every = 50

    mapping_optimizer = RiemannianSGD(mapping.parameters(), lr=learning_rate, rgrad=poincare_grad, retraction=retraction)
    training_pairs = {}
    for idx in indices:
        result = util.pairfromidx(idx, edge_folder)
        training_pairs[idx] = result

    logging.info(f"Started the run with lr {learning_rate}")

    for n in range(n_epochs):
        #print("Starting epoch "+str(n))
        #logging.info(f"Starting epoch {n}")
        iter=1
        
        if n!=0 and n%5==0:
            mapping_optimizer = RiemannianSGD(mapping.parameters(), lr=learning_rate*0.5, rgrad=poincare_grad, retraction=retraction)
        for i in indices:
            input_matrix = euclidean_embeddings[i]
            target_matrix = training_pairs[i][1]
            size = training_pairs[i][2]
            G = training_pairs[i][3]
            #print("graph size = ", G.order())

            if (iter-1) % resample_every == 0:
                subsampled_rows = np.random.permutation(G.order())[:subsample_row_num]   
                #print("RESAMPLED!") 

            loss, edge_acc = trainFCHyp(input_matrix, target_matrix, G, size, mapping, subsampled_rows, mapping_optimizer)
            print_loss_total += loss
            plot_loss_total += loss

            if True: #iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                #print('%s (%d %d%%) %.4f' % (util.timeSince(start, iter / n_iters),
                                            #iter, iter / n_iters * 100, print_loss_avg))
                logging.info('%s (%d %d%%) %.4f' % (util.timeSince(start, iter / n_iters),
                                            n, iter / n_iters * 100, print_loss_avg))

            iter+=1

        # do a full pass at the end:
        if n % full_stats_every == 0 or n == n_epochs-1:
            full_distortion = 0

            for i in indices:
                input_matrix = euclidean_embeddings[i]
                target_matrix = training_pairs[i][1]
                size = training_pairs[i][2]
                G = training_pairs[i][3]
                [fd, ea] = full_stats_pass_point(input_matrix, target_matrix, G, size, mapping)

                full_distortion += fd
                edge_acc        += ea

            print("\nFull distortion = ", full_distortion / len(indices), " Edge Acc. = ", edge_acc, "\n")

    #evaluation
    with torch.no_grad():
        testpairs = util.gettestpairs(test_folder)
        for name in testpairs.keys():
            [euclidean_emb, ground_truth, target_tensor, n] = testpairs[name]
            output = mapping(euclidean_emb.float())
            dist_recovered = util.distance_matrix_hyperbolic(output)
            loss = util.distortion(target_tensor, dist_recovered, n)
            logging.info(f"Testing on: {name}")
            print("distortion", loss)
            logging.info(f"Distortion: {loss}")
            dummy = dist_recovered.clone()
            edge_acc = util.compare_mst(ground_truth, dummy.detach().cpu().numpy())
            print("edge acc", edge_acc)
            logging.info(f"Edge accuracy: {edge_acc}")




@argh.arg("run_name", help="Directory to store the run")

def run(run_name, edge_folder="./data/edges/test_small_trees/", test_folder="./random_trees/test/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    #Set up logging.
    formatter = logging.Formatter('%(asctime)s %(message)s')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        datefmt='%FT%T',)
    logging.info(f"Logging to {run_name}")
    log = logging.getLogger()
    fh  = logging.FileHandler(run_name+"/log")
    fh.setFormatter(formatter)
    log.addHandler(fh)
    print("Running hMDS")
    run_hmds(run_name)

    logging.info(device)
    print("Running Net")
    run_net(run_name, edge_folder, test_folder, device)


if __name__ == '__main__':

    _parser = argh.ArghParser()
    _parser.set_default_command(run)
    _parser.dispatch()
