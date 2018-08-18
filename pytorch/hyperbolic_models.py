import numpy as np
import torch
from torch import nn

from hyperbolic_parameter import Hyperbolic_Parameter
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

### Embedding Models
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
    # machine_eps = np.finfo(uu.data.numpy().dtype).eps  # problem with cuda tensor
    # return acosh(torch.clamp(uu, min=1+machine_eps))
    return acosh(uu)

def h_proj(x, eps=1e-9):
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

class Hyperbolic_Emb(nn.Module):
    def __init__(self, n, d, project=True, initialize=None, learn_scale=False, absolute_loss=False, exponential_rescale=None):
        super().__init__()
        self.n = n
        #self.pairs     = n*(n-1)/2.
        self.pairs     = n # Due to sampling, we may not be prop to n/2

        self.H = Embedding(n, d, project, initialize, learn_scale)

        self.absolute_loss = absolute_loss
        abs_str = "absolute" if self.absolute_loss else "relative"

        self.exponential_rescale = exponential_rescale
        exp_str = f"Exponential {self.exponential_rescale}" if self.exponential_rescale is not None else "No Rescale"
        logging.info(f"{abs_str} {exp_str}")

    def step_rescale( self, values ):
        y = cudaify( torch.ones( values.size() ).double()/(10*self.n) )
        y[torch.lt( values.data, 5)] = 1.0
        return Variable(y, requires_grad=False)
        #return values**(-2)

    def scale(self):
        return self.H.scale()

    def dist(self, idx):
        return self.H.dist(idx)
    def dist_row(self, i):
        return self.H.dist_row(i)
    def dist_matrix(self):
        return self.H.dist_matrix()

    def loss(self, _x):
        idx, values = _x
        d = self.H.dist(idx)

        #term_rescale = torch.exp( 2*(1.-values) ) if self.exponential_rescale else self.step_rescale(values)
        term_rescale  = torch.exp( self.exponential_rescale*(1.-values) ) if self.exponential_rescale is not None else 1.0
        if self.absolute_loss:
            return torch.sum( term_rescale*( d - values)**2) /self.pairs
        else:
            return torch.sum( term_rescale*(d/values - 1)**2 ) / self.pairs

    def normalize(self):
        self.H.normalize()


class Embedding(nn.Module):
    def __init__(self, n, d, project=True, initialize=None, learn_scale=False):
        super().__init__()
        self.d = d
        self.project   = project
        if initialize is not None: logging.info(f"Initializing {np.any(np.isnan(initialize.numpy()))} {initialize.size()} {(n,d)}")
        x      = h_proj( 1e-3 * torch.rand(n, d).double() ) if initialize is None  else torch.DoubleTensor(initialize[0:n,0:d])
        self.w = Hyperbolic_Parameter(x)
        z =  torch.tensor([0.0], dtype=torch.double)
        if learn_scale:
            self.scale_log       = nn.Parameter(torch.tensor([0.0], dtype=torch.double))
            self.scale_log.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
        else:
            self.scale_log       = torch.tensor([0.0], dtype=torch.double, device=device)
        self.learn_scale = learn_scale
        self.scale_clamp       = 3.0
        logging.info(f"{self} {torch.norm(self.w.data - x)} {x.size()}")

    def scale(self):
        # print(self.scale_log.type(), self.lo_scale.type(), self.hi_scale.type())
        # scale = torch.exp(torch.clamp(self.scale_log, -self.thres, self.thres))
        scale = torch.exp(self.scale_log.tanh()*self.scale_clamp)
        # scale = scale if self.learn_scale else 1.0
        return scale

    def dist(self, idx):
        # print("idx shape: ", idx.size(), "values shape: ", values.size())
        wi = torch.index_select(self.w, 0, idx[:,0])
        wj = torch.index_select(self.w, 0, idx[:,1])
        d = dist(wi,wj)
        # print("loss: scale ", self.scale.data)
        return d / self.scale() # rescale to the size of the true distances matrix
        # return dist(wi,wj)*(1+self.scale)

    def dist_row(self, i):
        m = self.w.size(0)
        return dist(self.w[i,:].clone().unsqueeze(0).repeat(m,1), self.w) / self.scale()
        # return (1+self.scale)*dist(self.w[i,:].clone().unsqueeze(0).repeat(m,1), self.w)

    def dist_matrix(self):
        m    = self.w.size(0)
        rets = torch.zeros(m, m, dtype=torch.double, device=device)
        for i in range(m):
            rets[i,:] = self.dist_row(i)
        return rets

    def normalize(self):
        if self.project:
            self.w.proj()
            # print("normalize: scale ", self.scale.data)
            # print(type(self.scale_log), self.scale_log.type())
            # self.scale_log = torch.clamp(self.scale_log, self.lo_scale, self.hi_scale)
