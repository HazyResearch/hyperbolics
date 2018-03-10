import logging, argh
import sys
# Data prep.
#import utils.data_prep as data_prep
import utils.load_graph as load_graph
import networkx as nx
import scipy
import scipy.sparse.csgraph as csg
import utils.distortions as dis
import graph_helpers as gh
import mds_warmstart
import pandas
from hyperbolic_parameter import Hyperbolic_Parameter
from hyperbolic_models import Hyperbolic_Emb, dist

# This describes a hyperbolic optimizer in Pytorch. It requires two modifications:
# 
# * When declaring a parameter, one uses a class called "Hyperbolic
# * Parameter". It assumes that the _last_ dimension is in the
# * disk. E.g., a tensor of size n x m x d means that you have n x m
# * elements of H_D. d >= 2.
# 
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

# compute marix of non-squard distances
def dist_matrix(_data):
    m    = _data.size(0)
    rets = cudaify( torch.DoubleTensor(m, m).zero_() )
    for i in range(m):
        rets[i,:] = dist(_data[i,:].clone().unsqueeze(0).repeat(m,1), _data)
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
    hm.train(True)
    opt.zero_grad()
    loss = hm.loss(cu_var(data, requires_grad=False))
    loss.backward()
    Hyperbolic_Parameter.correct_metric(hm.parameters()) # NB: THIS IS THE NEW CALL
    opt.step()
    # Projection
    hm.normalize()
    return loss


class GraphRowSubSampler(torch.utils.data.Dataset):
    def __init__(self, G, scale, subsample, Z=None):
        super(GraphRowSubSampler, self).__init__()
        self.graph     = nx.to_scipy_sparse_matrix(G)
        self.n         = G.order()
        self.scale     = scale
        self.subsample = subsample
        self.val_cache = torch.DoubleTensor(self.n,subsample).zero_()
        self.idx_cache = torch.LongTensor(self.n,subsample,2).zero_()
        self.cache     = set()
        self.verbose   = False
        self.n_cached  = 0
        self.Z         = Z
        logging.info(self)
        
    def __getitem__(self, index):
        if index not in self.cache:
            if self.verbose: logging.info(f"Cache miss for {index}")
            h = gh.djikstra_wrapper( (self.graph, [index]) )[0,:] if self.Z is None else self.Z[index,:]
            # add in all the edges
            cur = 0
            self.idx_cache[index,:,0] = index
            neighbors = scipy.sparse.find(self.graph[index,:])[1]
            for e in neighbors:
                self.idx_cache[index,cur,1] = int(e)
                self.val_cache[index,cur] = self.scale
                cur += 1
                if cur >= self.subsample: break
            
            scratch   = np.array(range(self.n))
            np.random.shuffle(scratch)

            i = 0
            while cur < self.subsample and i < self.n:
                v = scratch[i]
                if v != index and v not in neighbors:
                    self.idx_cache[index,cur,1] = int(v)
                    self.val_cache[index,cur]   = self.scale*h[v]
                    cur += 1
                i += 1
            if self.verbose: logging.info(f"\t neighbors={neighbors} {self.idx_cache[index,:,1].numpy().T}")
            self.cache.add(index)
            self.n_cached += 1
            if self.n_cached % (max(self.n//20,1)) == 0: logging.info(f"\t Cached {self.n_cached} of {self.n}")
            
        return (self.idx_cache[index,:], self.val_cache[index,:])
    
    def __len__(self): return self.n

    def __repr__(self):
        return f"Subsample: {self.n} points with scale {self.scale} subsample={self.subsample}"
            

class GraphRowSampler(torch.utils.data.Dataset):
    def __init__(self, G, scale, use_cache=True):
        self.graph = nx.to_scipy_sparse_matrix(G)
        self.n     = G.order()
        self.scale = scale
        self.cache = dict() if use_cache else None
            
    def __getitem__(self, index):
        h = None
        if self.cache is None or index not in self.cache:
            h = gh.djikstra_wrapper( (self.graph, [index]) )
            if self.cache is not None:
                self.cache[index] = h
            #logging.info(f"info {index}")
        else:
            h = self.cache[index]
            #logging.info(f"hit {index}")
            
        idx = torch.LongTensor([ (index, j) for j in range(self.n) if j != index])
        v   = torch.DoubleTensor(h).view(-1)[idx[:,1]]        
        return (idx, v)
    
    def __len__(self): return self.n

    def __repr__(self):
        return f"DATA: {self.n} points with scale {self.scale}"
    
#
# DATA Diagnostics
#
def major_stats(G, scale, n, m, lazy_generation, Z,z, n_rows_sampled=250, num_workers=16):
    m.train(False)
    if lazy_generation:
        logging.info(f"\t Computing Major Stats lazily... ")
        avg, me, mc = 0.0, 0.0, 0.0
        good,bad    = 0,0
        _count      = 0 
        for u in z:
            index,vs = u
            v_rec  = m.dist(cu_var(index)).data.cpu().numpy()
            v      = vs.cpu().numpy()
            for i in range(len(v)):
                if dis.entry_is_good(v[i], v_rec[i]):
                    (_avg,me,mc) = dis.distortion_entry(v[i]*scale, v_rec[i], me, mc)
                    avg         += _avg
                    good        += 1
                else:
                    bad         += 1
            _count += len(v)
            if n_rows_sampled*n < _count:
                logging.info(f"\t\t Completed {n} {n_rows_sampled} {_count} good={good} bad={bad}") 
                break
        avg_dist     = avg/good
        dist_max     = me
        nan_elements = bad
        map_avg      = 0.0
        
        # sample for rows
        shuffled     = list(range(n))
        #np.random.shuffle(shuffled)
        mm = 0
        for i in shuffled[0:n_rows_sampled]:
            h_rec      = m.dist_row(i).cpu().data.numpy()
            map_avg   += dis.map_via_edges(G,i, h_rec)
            mm        += 1         
        mapscore = map_avg/mm
    else:
        #H    = Z/scale
        H    = Z*scale
        Hrec = dist_matrix(m.w.data).cpu().numpy()
        logging.info("Compare matrices built")  
        dist_max, avg_dist, nan_elements = dis.distortion(H, Hrec, n, num_workers)
        mapscore = dis.map_score(H, Hrec, n, num_workers) 
        
    logging.info(f"Distortion avg={avg_dist} wc={dist_max} nan_elements={nan_elements}")  
    logging.info(f"MAP = {mapscore}")   
    logging.info(f"data_scale={scale} scale={m.scale.data[0]}")
        
                                
@argh.arg("dataset", help="dataset number")
@argh.arg("-r", "--rank", help="Rank to use")
@argh.arg("-s", "--scale", help="Scale factor")
@argh.arg("-l", "--learning-rate", help="Learning rate")
@argh.arg("-t", "--tol", help="Tolerances for projection")
@argh.arg("-y", "--use-yellowfin", help="Turn off yellowfin")
@argh.arg("--epochs", help="number of steps in optimization")
@argh.arg("--print-freq", help="print loss this every this number of steps")
@argh.arg("--model-save-file", help="Save model file")
@argh.arg("--model-load-file", help="Load model file")
@argh.arg("--batch-size", help="Batch size")
@argh.arg("--num-workers", help="Number of workers for loading. Default is to use all cores")
@argh.arg("-g", "--lazy-generation", help="Use a lazy data generation technique")
@argh.arg("--subsample", type=int, help="Number of edges to subsample")
@argh.arg("--log-name", help="Log to a file")
@argh.arg("--force-sgd", help="Force using plain SGD")
@argh.arg("-w", "--warm-start", help="Warm start the model with MDS")
@argh.arg("--learn-scale", help="Learn scale")
@argh.arg("--sample", help="Sample the distance matrix")
@argh.arg("--checkpoint-freq", help="Checkpoint Frequency (Expensive)")
@argh.arg("-e", "--exponential-rescale", type=float, help="Exponential Rescale")
@argh.arg("-x", "--extra-steps", type=int, help="Steps per batch")
@argh.arg("--use-svrg", help="Use SVRG")
@argh.arg("-T", help="SVRG T parameter")
def learn(dataset, rank=2, scale=1., learning_rate=1e-1, tol=1e-8, epochs=100,
          use_yellowfin=False, force_sgd=False, print_freq=1, model_save_file=None, model_load_file=None, batch_size=16,
          num_workers=None, lazy_generation=False, log_name=None, warm_start=None, learn_scale=False, checkpoint_freq=1000, sample=1., subsample=None, 
          exponential_rescale=None, extra_steps=1, use_svrg=False, T=10):
    # Log configuration
    formatter = logging.Formatter('%(asctime)s %(message)s')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%FT%T',)
    if log_name is not None:
        logging.info(f"Logging to {log_name}")
        log = logging.getLogger()
        fh  = logging.FileHandler(log_name)
        fh.setFormatter( formatter )
        log.addHandler(fh)

    logging.info(f"Commandline {sys.argv}")
    if model_save_file is None: logging.warn("No Model Save selected!")
    G  = load_graph.load_graph(dataset)
    GM = nx.to_scipy_sparse_matrix(G)

    n = G.order()
    logging.info(f"Loaded Graph {dataset} with {n} nodes scale={scale}")

    Z = None
    
    def collate(ls):
        x, y = zip(*ls)
        return torch.cat(x), torch.cat(y)
    
    if lazy_generation:
        if subsample is not None:
            z = DataLoader(GraphRowSubSampler(G, scale, subsample), batch_size, shuffle=True, collate_fn=collate)
        else:
            z = DataLoader(GraphRowSampler(G, scale), batch_size, shuffle=True, collate_fn=collate)
        logging.info("Built Data Sampler")
    else:
        Z   = gh.build_distance(G, scale, num_workers=int(num_workers) if num_workers is not None else 16)   # load the whole matrix    
        logging.info(f"Built distance matrix with {scale} factor")

        if subsample is not None:
            z = DataLoader(GraphRowSubSampler(G, scale, subsample,Z=Z), batch_size, shuffle=True, collate_fn=collate)
        else:
            idx       = torch.LongTensor([(i,j)  for i in range(n) for j in range(i+1,n)])
            Z_sampled = gh.dist_sample_rebuild_pos_neg(Z, sample) if sample < 1 else Z
            vals      = torch.DoubleTensor([Z_sampled[i,j] for i in range(n) for j in range(i+1, n)])
            z         = DataLoader(TensorDataset(idx,vals), batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
        logging.info("Built data loader")
    
  
    if model_load_file is not None:
        logging.info(f"Loading {model_load_file}...")
        m = cudaify( torch.load(model_load_file) )
        logging.info(f"Loaded scale {m.scale.data[0]} {torch.sum(m.w.data)} {m.epoch}")
    else:
        logging.info(f"Creating a fresh model warm_start?={warm_start}")
        # m_init = torch.DoubleTensor(mds_warmstart.get_normalized_hyperbolic(mds_warmstart.get_model(int(dataset),rank)[1])) if warm_start else None

        # load from DataFrame; assume that the julia code has been called prior and saved in "savefile"
        if warm_start is not None:
            # print(type(mds_warmstart.get_model(int(dataset),rank)[1]))
            m_init = pandas.read_csv(warm_start, index_col=0).as_matrix()
            # m_init = torch.DoubleTensor(mds_warmstart.get_normalized_hyperbolic(m_init))
            m_init = torch.DoubleTensor(m_init)
        else:
            m_init = None
        # print(m_init)
        logging.info(f"\t Warmstarting? {warm_start} {m_init.size() if warm_start else None} {G.order()}")

        m       = cudaify( Hyperbolic_Emb(G.order(), rank, initialize=m_init, learn_scale=learn_scale, exponential_rescale=exponential_rescale) )
        m.normalize()
        m.epoch = 0
    logging.info(f"Constructed model with rank={rank} and epochs={m.epoch} isnan={np.any(np.isnan(m.w.cpu().data.numpy()))}")

    #
    # Build the Optimizer
    #
    # TODO: Redo this in a sensible way!!
    #
    if force_sgd: opt = torch.optim.SGD(m.parameters(), lr=learning_rate)
    else:
        from yellowfin import YFOptimizer
        opt = YFOptimizer(m.parameters()) if use_yellowfin else torch.optim.Adagrad(m.parameters()) 
    
    if use_svrg:
        from svrg import SVRG
        base_opt = torch.optim.SGD if force_sgd else torch.optim.Adagrad
        opt      = SVRG(m.parameters(), lr=learning_rate, T=T, data_loader=z, opt=base_opt)
   
        
    logging.info(opt)

    for i in range(m.epoch, m.epoch+epochs):
        l = 0.0
        m.train(True)
        if use_svrg:
            for data in z:
                def closure(data=data, target=None):
                    _data = data if target is None else (data,target)
                    c = m.loss(cu_var(_data))  
                    c.backward()
                    return c.data[0]
                l += opt.step(closure)
                # Projection
                m.normalize()

        else:
            opt.zero_grad() # This is handled by the SVRG.
            for the_step in range(extra_steps):
                # Accumulate the gradient
                for u in z:                
                    _loss = m.loss(cu_var(u, requires_grad=False))
                    _loss.backward()
                    l += _loss.data[0]
                Hyperbolic_Parameter.correct_metric(m.parameters()) # NB: THIS IS THE NEW CALL
                opt.step()
                # Projection
                m.normalize()
                
                #l += step(m, opt, u).data[0]

        # Logging code
        if l < tol:
                logging.info("Found a {l} solution. Done at iteration {i}!")
                break
        if i % print_freq == 0:
            logging.info(f"{i} loss={l}")
        if i % checkpoint_freq == 0:
            logging.info(f"\n*** Major Checkpoint. Computing Stats and Saving")
            major_stats(GM,1+m.scale.data[0],n,m, True, Z, z)
            if model_save_file is not None:
                fname = f"{model_save_file}.{m.epoch}"
                logging.info(f"Saving model into {fname} {torch.sum(m.w.data)} ") 
                torch.save(m, fname)
            logging.info("*** End Major Checkpoint\n")
        m.epoch += 1
        
    logging.info(f"final loss={l}")

    if model_save_file is not None:
        fname = f"{model_save_file}.final"
        logging.info(f"Saving model into {fname}-final {torch.sum(m.w.data)} {m.scale.data[0]}")
        torch.save(m, fname)

    major_stats(GM,1+m.scale.data[0], n, m, True, Z,z)

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    _parser.add_commands([learn])
    _parser.dispatch()
