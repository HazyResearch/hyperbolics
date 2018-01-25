import logging, argh
# Data prep.
import data_prep
import networkx as nx
import scipy.sparse.csgraph as csg
# This describes a hyperbolic optimizer in Pytorch. It requires two modifications:
# 
# * When declaring a parameter, one uses a class called "Hyperbolic Parameter". It assumes that the _last_ dimension is in the disk. E.g., a tensor of size n x m x d means that you have n x m elements of H_D. d >= 2.
#   * It inherits from parameter, so you can do anything you want with it (set its value, pass it around).
# 
# * So that you can use any optimizer and get the correction, after the `backward` call but before the `step`, you need to call a function called `hyperbolic_fix`. It will walk the tree, look for the hyperbolic parameters and correct them. 
#  * The step function below can be used pretty generically.


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import numpy as np, math
import random
from hyperbolic_parameter import Hyperbolic_Parameter

# We implement both in pytorch using a custom SGD optimizer. This is used to correct for the hyperbolic variables.
# 
# Here are the basic distance and projection functions. The distance in Poincaré space is:
# 
# $$ d(u,v) = \mathrm{arcosh}\left(1 + 2\frac{\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$$
# 
# We implement a simple projection on the disk as well.


#{\displaystyle \operatorname {arcosh} x=\ln \left(x+{\sqrt {x^{2}-1}}\right)}
def acosh(x):
    return torch.log(x + torch.sqrt(x**2-1))

def dist(u,v):
    z  = 2*torch.norm(u-v,2,1)**2
    uu = 1. + torch.div(z,((1-torch.norm(u,2,1)**2)*(1-torch.norm(v,2,1)**2)))
    return acosh(uu)

#
# Differs from the Facebook paper, slightly.
#
def h_proj(x, eps=1e-5):
    current_norms = torch.norm(x,2,x.dim() - 1)
    mask_idx      = current_norms < 1.0
    modified      = 1./((1+eps)*current_norms)
    modified[mask_idx] = 1.0
    new_size      = [1]*current_norms.dim() + [x.size(x.dim()-1)]
    modified      = modified.unsqueeze(modified.dim()).repeat(*new_size) 
    return x * modified

def dot(x,y): return torch.sum(x * y, 1)

# Compute the  
# $$\min_{v} \sum_{j=1}^{n} \mathrm{acosh}\left(1 + d^2_E(L(v), w_j)\right)^2$$
def line_dist_sq(_x,y):
    norm_x = torch.norm(_x)**(-2)
    x = _x.repeat(y.size(0),1)
    return torch.norm(y - torch.diag(dot(x,y)*norm_x)@x,2,1)**2

#
# Our models
# 
class Hyperbolic_Mean(nn.Module):
    def __init__(self, d):
        super(Hyperbolic_Mean, self).__init__()
        self.w = Hyperbolic_Parameter( (torch.rand(d) * 1e-3).double() ) 
            
    def loss(self, y_data): 
        return torch.sum(dist(self.w.repeat(y_data.size(0),1),y_data)**2)
 
    def normalize(self):
        self.w.proj()

class Hyperbolic_Lines(nn.Module):
    def __init__(self, d):
        super(Hyperbolic_Lines, self).__init__()
        self.w = Hyperbolic_Parameter(h_proj(torch.rand(d) * 1e-3).double()) 
            
    # $$\min_{v} \sum_{j=1}^{n} \mathrm{acosh}\left(1 + d^2_E(L(v), w_j)\right)^2$$
    # learn the lines in a zero centered way.
    def loss(self, y_data): 
        return torch.sum(acosh(1 + line_dist_sq(self.w, y_data))**2)
    
    def normalize(self): # we handle this in the line_dist_s
        return

class Hyperbolic_Emb(nn.Module):
    def __init__(self, n, d, project=True):
        super(Hyperbolic_Emb, self).__init__()
        self.n = n
        self.d = d
        self.pairs = n*(n-1)/2. 
        self.project   = project
        self.w = Hyperbolic_Parameter(h_proj( 1e-3 * torch.rand(n, d).double() ) )
        
    def loss(self, _x):
        idx, values = _x
        wi = torch.index_select(self.w, 0, idx[:,0])
        wj = torch.index_select(self.w, 0, idx[:,1])
        return torch.sum((dist(wi,wj) - values)**2)/self.pairs

    def normalize(self):
        if self.project: self.w.proj()


# compute marix of squard distances
def dist_matrix(_data):
    m    = _data.size(0)
    rets = torch.DoubleTensor(m, m).zero_()
    for i in range(m):
        xx = dist(_data[i,:].clone().unsqueeze(0).repeat(m,1), _data)**2
        for j in range(m):
            rets[i,j] = xx[j]
    return rets


#
# This is the step for training
#
def cu_var(x, requires_grad=None, volatile=False):
    if isinstance(x, list) : return [cu_var(u, requires_grad=requires_grad, volatile=volatile) for u in x]
    if isinstance(x, tuple): return tuple([cu_var(u, requires_grad=requires_grad, volatile=volatile) for u in list(x)])
    rg = not volatile if requires_grad is None else requires_grad
    vx = Variable(x, requires_grad=rg, volatile=volatile)
    return vx.cuda() if torch.cuda.is_available() else vx  
def cudaify(x):            return x.cuda() if torch.cuda.is_available() else x

def step(hm, opt, data):
    opt.zero_grad()
    loss = hm.loss(cu_var(data, requires_grad=False))
    loss.backward()
    Hyperbolic_Parameter.correct_metric(hm.parameters()) # NB: THIS IS THE NEW CALL
    opt.step()
    # Projection
    hm.normalize()
    return loss


def example():
    learning_rate  = 1e-3
    tol            = 1e-8
    hl             = Hyperbolic_Lines(d)
    opt_hl         = torch.optim.SGD(hl.parameters(), lr=learning_rate)
    line_norm      = (1-torch.norm(_data_z,2,1)**2)**(-2)
    line_data      = torch.diag(line_norm)@_data_z
    for t in range(1001):
        l2 = step(hl, opt_hl, line_data)
        ll = l2.data[0]
        if ll < tol: 
            logging.info(f"End {t} {ll}")
            break
        if t % 100 == 0:
            logging.info(f"{t:2} -> {ll:0.2f}")


# Should be moved to utility
from multiprocessing import Pool

def djikstra_wrapper( _x ):
    (mat, x) = _x
    return csg.dijkstra(mat, indices=x, unweighted=True, directed=False)
def build_distance(G, scale, num_workers=None):
    n = G.order()
    p = Pool() if num_workers is None else Pool(num_workers)
    
    adj_mat_original = nx.to_scipy_sparse_matrix(G)

    # Simple chunking
    nChunks     = 128
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
        H = djisktra_wrapper(adj_mat_original, range(n))
        
    H *= scale
    return H

def build_distance_hyperbolic(G, scale):
    return np.cosh(build_distance(G,scale)-1.)/2.


@argh.arg("dataset", help="dataset number")
@argh.arg("-r", "--rank", help="Rank to use")
@argh.arg("-s", "--scale", help="Scale factor")
@argh.arg("-l", "--learning-rate", help="Learning rate")
@argh.arg("-t", "--tol", help="Tolerances for projection")
@argh.arg("-y", "--use-yellowfin", help="Turn off yellowfin")
@argh.arg("--epochs", help="number of steps in optimization")
@argh.arg("--print-freq", help="print loss this every this number of steps")
@argh.arg("--model-save-file", help="Save model file")
@argh.arg("--batch-size", help="Batch size")
@argh.arg("--num-workers", help="Number of workers for loading. Default is to use all cores")
def learn(dataset, rank=2, scale=1., learning_rate=1e-3, tol=1e-8, epochs=100,
          use_yellowfin=False, print_freq=1, model_save_file=None, batch_size=16, num_workers=None):
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s.%(msecs)03d" + " learn " + "%(levelname)s %(name)s: %(message)s",
                        datefmt='%FT%T',)

    if model_save_file is None: logging.warn("No Model Save selected!")
    G = data_prep.load_graph(int(dataset))
    n = G.order()
    logging.info(f"Loaded Graph {dataset} with {n} nodes")
    
    Z   = build_distance(G, scale, num_workers=num_workers)   # load the whole matrix

    
    logging.info(f"Built distance matrix with {scale} factor")
    idx  = torch.LongTensor([(i,j)  for i in range(n) for j in range(i+1,n)])
    vals = torch.DoubleTensor([Z[i,j] for i in range(n) for j in range(i+1, n)])
    z  = DataLoader(TensorDataset(idx,vals), batch_size=batch_size, shuffle=True)
    logging.info("Built data loader")

    m   = cudaify( Hyperbolic_Emb(G.order(), rank) )
    logging.info(f"Constucted model with rank={rank}")

    from yellowfin import YFOptimizer
    opt = YFOptimizer(m.parameters()) if use_yellowfin else torch.optim.SGD(m.parameters(), lr=learning_rate)
    
    for i in range(epochs):
        l = 0.0
        for u in z:
            l += step(m, opt, u).data[0]
            
        if l < tol:
                logging.info("Found a {l} solution. Done at iteration {i}!")
                break
        if i % print_freq == 0:
            logging.info(f"{i} loss={l}")
    logging.info(f"final loss={l}")

    # TODO: PRINT Data diagnostics
    logging.warn("Call data diagnoistics function!")
    
    if model_save_file is not None: torch.save(m, model_save_file)

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    _parser.add_commands([learn])
    _parser.dispatch()
