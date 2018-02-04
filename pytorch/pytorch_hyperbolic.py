import logging, argh
# Data prep.
import data_prep
import networkx as nx
import scipy.sparse.csgraph as csg
import distortions as dis
import graph_helpers as gh
import mds_warmstart 
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
    def __init__(self, n, d, project=True, initialize=None, learn_scale=False, absolute_loss=False):
        super(Hyperbolic_Emb, self).__init__()
        self.n = n
        self.d = d
        self.pairs     = n*(n-1)/2. 
        self.project   = project
        if initialize is not None: logging.info(f"Initializing {np.any(np.isnan(initialize.numpy()))} {initialize.size()} {(n,d)}")
        x      = h_proj( 1e-3 * torch.rand(n, d).double() ) if initialize is None  else torch.DoubleTensor(initialize[0:n,0:d])
        self.w = Hyperbolic_Parameter(x)
        self.scale       = nn.Parameter( torch.DoubleTensor([0.0]))
        self.learn_scale = learn_scale
        self.lo_scale    = -0.99
        self.hi_scale    = 10.0
        self.absolute_loss = absolute_loss
        abs_str = "absolute" if self.absolute_loss else "relative"

        self.exponential_rescale = True
        exp_str = "exponential" if self.exponential_rescale else "Step Rescale"
        logging.info(f"{torch.norm(self.w.data - x)} {x.size()} {abs_str} {exp_str}")
        logging.info(self)

    def step_rescale( self, values ):
        y = cudaify( torch.ones( values.size() ).double()/(10*self.n) )
        y[torch.lt( values.data, 5)] = 1.0
        return Variable(y, requires_grad=False)
        #return values**(-2)

    def loss(self, _x):
        idx, values = _x
        wi = torch.index_select(self.w, 0, idx[:,0])
        wj = torch.index_select(self.w, 0, idx[:,1])
        _scale = 1+torch.clamp(self.scale,self.lo_scale,self.hi_scale)

        term_rescale = torch.exp( 2*(1.-values) ) if self.exponential_rescale else self.step_rescale(values) 
        if self.absolute_loss:
            _values = values*_scale if self.learn_scale else values 
            return torch.sum( term_rescale*( dist(wi,wj) - _values))**2/self.pairs 
        else:
            _s = _scale if self.learn_scale else 1.0
            return torch.sum( term_rescale*_s*(dist(wi,wj)/values - 1.0)**2/self.pairs )
        
    def normalize(self):
        if self.project: 
            self.w.proj()
            self.scale.data = torch.clamp(self.scale.data,self.lo_scale, self.hi_scale)
        


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



# 
class GraphRowSampler(torch.utils.data.Dataset):
    def __init__(self, G, scale):
        self.graph = nx.to_scipy_sparse_matrix(G)
        self.n     = G.order()
        self.scale = scale
        logging.info(f"{type(self.graph)}")
        
    def __getitem__(self, index):
        h   = gh.djikstra_wrapper( (self.graph, [index]) )
        idx = torch.LongTensor([ (index, j) for j in range(self.n) if j != index])
        v   = torch.DoubleTensor( [ h[0,j] for j in range(self.n) if j != index] )
        return (idx, v)
    
    def __len__(self): return self.n

    def __repr__(self):
        return f"DATA: {self.n} points with scale {self.scale}"
    
#
# DATA Diagnostics
#
def major_stats(G, scale, n, m, lazy_generation, Z):
    m.train(False)
    H    = gh.build_distance(G, 1.0, num_workers=1) if lazy_generation else Z/scale
    Hrec = dist_matrix(m.w.data).cpu().numpy()
    logging.info("Compare matrices built")  
    dist_max, avg_dist, nan_elements = dis.distortion(H, Hrec, n, 2)
    logging.info(f"Distortion avg={avg_dist} wc={dist_max} nan_elements={nan_elements}")  
    mapscore = dis.map_score(H, Hrec, n, 2) 
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
@argh.arg("--load-model-file", help="Load model file")
@argh.arg("--batch-size", help="Batch size")
@argh.arg("--num-workers", help="Number of workers for loading. Default is to use all cores")
@argh.arg("-g", "--lazy-generation", help="Use a lazy data generation technique")
@argh.arg("--log-name", help="Log to a file")
@argh.arg("--use-sgd", help="Force using plan SGD")
@argh.arg("-w", "--warm-start", help="Warm start the model with MDS")
@argh.arg("--learn-scale", help="Learn scale")
@argh.arg("--checkpoint-freq", help="Checkpoint Frequency (Expensive)")
def learn(dataset, rank=2, scale=1., learning_rate=1e-2, tol=1e-8, epochs=100,
          use_yellowfin=False, use_sgd=True, print_freq=1, model_save_file=None, load_model_file=None, batch_size=16,
          num_workers=None, lazy_generation=False, log_name=None, warm_start=False, learn_scale=False, checkpoint_freq=1000):
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

    if model_save_file is None: logging.warn("No Model Save selected!")
    G = data_prep.load_graph(int(dataset))
    n = G.order()
    logging.info(f"Loaded Graph {dataset} with {n} nodes scale={scale}")

    Z = None
    if lazy_generation:
        def collate(ls):
            x, y = zip(*ls)
            return torch.cat(x), torch.cat(y)
        z = DataLoader(GraphRowSampler(G, scale), batch_size, shuffle=True, collate_fn=collate)
        logging.info("Built data Sampler")
    else:
        Z   = gh.build_distance(G, scale, num_workers=num_workers)   # load the whole matrix    
        logging.info(f"Built distance matrix with {scale} factor")
        idx  = torch.LongTensor([(i,j)  for i in range(n) for j in range(i+1,n)])
        vals = torch.DoubleTensor([Z[i,j] for i in range(n) for j in range(i+1, n)])
        z  = DataLoader(TensorDataset(idx,vals), batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
        logging.info("Built data loader")


    if load_model_file is not None:
        logging.info(f"Loading {load_model_file}...")
        m = cudaify( torch.load(load_model_file) )
        logging.info(f"Loaded scale {m.scale.data[0]} {torch.sum(m.w.data)}")
    else:
        m_init = torch.DoubleTensor(mds_warmstart.get_normalized_hyperbolic(mds_warmstart.get_model(int(dataset),rank)[1])) if warm_start else None
        logging.info(f"\t Warmstarting? {warm_start} {m_init.size() if warm_start else None} {G.order()}")

        m = cudaify( Hyperbolic_Emb(G.order(), rank, initialize=m_init, learn_scale=learn_scale) )
        m.epoch = 0
    logging.info(f"Constucted model with rank={rank} and epochs={m.epoch} isnan={np.any(np.isnan(m.w.cpu().data.numpy()))}")
    
                
    from yellowfin import YFOptimizer
    opt = YFOptimizer(m.parameters()) if use_yellowfin else torch.optim.Adagrad(m.parameters()) # 
    if use_sgd: opt = torch.optim.SGD(m.parameters(), lr=learning_rate)
    
    for i in range(epochs):
        l = 0.0
        for u in z:
            l += step(m, opt, u).data[0]
            
        if l < tol:
                logging.info("Found a {l} solution. Done at iteration {i}!")
                break
        if i % print_freq == 0:
            logging.info(f"{i} loss={l}")
        if i % checkpoint_freq == 0:
            logging.info(f"\n*** Major Checkpoint. Computing Stats and Saving")
            major_stats(G,scale,n,m, lazy_generation, Z)
            if model_save_file is not None:
                logging.info(f"Saving model into {model_save_file}-{m.epoch} {torch.sum(m.w.data)} ") 
                torch.save(m, f"{model_save_file}.{m.epoch}")
            logging.info("*** End Major Checkpoint\n")
        m.epoch += 1
        
    logging.info(f"final loss={l}")

    if model_save_file is not None:
        logging.info(f"Saving model into {model_save_file}-final {torch.sum(m.w.data)} {m.scale.data[0]}") 
        torch.save(m, f"{model_save_file}.final")

    major_stats(G,scale, n,m, lazy_generation, Z)

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    _parser.add_commands([learn])
    _parser.dispatch()
