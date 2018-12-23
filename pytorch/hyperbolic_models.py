import numpy as np
import torch
from torch import nn

from hyperbolic_parameter import PoincareParameter, EuclideanParameter, SphericalParameter, HyperboloidParameter
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#
# Our models
#
class Hyperbolic_Mean(nn.Module):
    def __init__(self, d):
        super(Hyperbolic_Mean, self).__init__()
        self.w = Hyperbolic_Parameter(sizes=d)

    def loss(self, y_data):
        return torch.sum(dist(self.w.repeat(y_data.size(0),1),y_data)**2)

    def normalize(self):
        self.w.proj()

class Hyperbolic_Lines(nn.Module):
    def __init__(self, d):
        super(Hyperbolic_Lines, self).__init__()
        self.w = Hyperbolic_Parameter(sizes=d)

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

# TODO: probably makes sense to move distance function into the corresponding Parameter type
def dist_p(u,v):
    z  = 2*torch.norm(u-v,2,1)**2
    uu = 1. + torch.div(z,((1-torch.norm(u,2,1)**2)*(1-torch.norm(v,2,1)**2)))
    # machine_eps = np.finfo(uu.data.numpy().dtype).eps  # problem with cuda tensor
    # return acosh(torch.clamp(uu, min=1+machine_eps))
    return acosh(uu)

# def h_proj(x, eps=1e-9):
#     current_norms = torch.norm(x,2,x.dim() - 1)
#     mask_idx      = current_norms < 1.0
#     modified      = 1./((1+eps)*current_norms)
#     modified[mask_idx] = 1.0
#     new_size      = [1]*current_norms.dim() + [x.size(x.dim()-1)]
#     modified      = modified.unsqueeze(modified.dim()).repeat(*new_size)
#     return x * modified

def dot(x,y): return torch.sum(x * y, -1)

def dist_e(u, v):
    """ Input shape (n, d) """
    return torch.norm(u-v, 2, dim=1)

def dist_s(u, v, eps=1e-9):
    uu = SphericalParameter._proj(u)
    vv = SphericalParameter._proj(v)
    return torch.acos(torch.clamp(dot(uu, vv), -1+eps, 1-eps))

# Compute the
# $$\min_{v} \sum_{j=1}^{n} \mathrm{acosh}\left(1 + d^2_E(L(v), w_j)\right)^2$$
def line_dist_sq(_x,y):
    norm_x = torch.norm(_x)**(-2)
    x = _x.repeat(y.size(0),1)
    return torch.norm(y - torch.diag(dot(x,y)*norm_x)@x,2,1)**2

class ProductEmbedding(nn.Module):
    def __init__(self, n, hyp_d, hyp_copies=1, euc_d=1, euc_copies=0, sph_d=1, sph_copies=0, project=True, initialize=None, learn_scale=False, initial_scale=0.0, absolute_loss=False, logrel_loss=False, dist_loss=False, square_loss=False, sym_loss=False, exponential_rescale=None, riemann=False):
        super().__init__()
        self.n = n
        self.riemann = riemann

        # self.H = nn.ModuleList([Embedding(dist_p, PoincareParameter, n, hyp_d, project, initialize, learn_scale, initial_scale) for _ in range(hyp_copies)])
        self.H = nn.ModuleList([Embedding(HyperboloidParameter.dist_h, HyperboloidParameter, n, hyp_d, project, initialize, learn_scale, initial_scale) for _ in range(hyp_copies)])
        self.E = nn.ModuleList([Embedding(dist_e, EuclideanParameter, n, euc_d, False, initialize, False, initial_scale) for _ in range(euc_copies)])
        # self.E = nn.ModuleList([Embedding(dist_e, EuclideanParameter, n, euc_d, False, initialize, learn_scale, initial_scale) for _ in range(euc_copies)])
        self.S = nn.ModuleList([Embedding(dist_s, SphericalParameter, n, sph_d, project, initialize, learn_scale, initial_scale) for _ in range(sph_copies)])

        self.scale_params = [H.scale_log for H in self.H] \
                          + [E.scale_log for E in self.E] \
                          + [S.scale_log for S in self.S] \
                          if learn_scale else []
        self.hyp_params = [H.w for H in self.H]
        self.euc_params = [E.w for E in self.E]
        self.sph_params = [S.w for S in self.S]
        self.embed_params = [H.w for H in self.H] \
                          + [E.w for E in self.E] \
                          + [S.w for S in self.S]

        self.absolute_loss = absolute_loss
        self.logrel_loss = logrel_loss
        self.dist_loss = dist_loss
        self.square_loss = square_loss
        self.sym_loss = sym_loss
        abs_str = "absolute" if self.absolute_loss else "relative"

        self.exponential_rescale = exponential_rescale
        exp_str = f"Exponential {self.exponential_rescale}" if self.exponential_rescale is not None else "No Rescale"
        logging.info(f"{abs_str} {exp_str}")

    def step_rescale( self, values ):
        y = cudaify( torch.ones( values.size() ).double()/(10*self.n) )
        y[torch.lt( values.data, 5)] = 1.0
        return Variable(y, requires_grad=False)
        #return values**(-2)

    def all_attr(self, fn):
        H_attr = [fn(H) for H in self.H]
        E_attr = [fn(E) for E in self.E]
        S_attr = [fn(S) for S in self.S]
        return H_attr + E_attr + S_attr

    def embedding(self):
        """ Return list of all entries of the embedding(s) """
        return torch.cat(self.all_attr(lambda emb: emb.w.view(-1)))
        # return torch.stack([H.w for H in self.H], dim=0)
        # return torch.stack(self.all_attr(lambda emb: emb.w), dim=0)
        # return (torch.stack([H.w for H in self.H], dim=0), torch.stack([H.w for H in self.E], dim=0), torch.stack([H.w for H in self.S], dim=0))

    def scale(self):
        # return [H.scale() for H in self.H]
        return self.all_attr(lambda emb: emb.scale())

    def dist_idx(self, idx):
        # return sum([H.dist_idx(idx) for H in self.H])
        d = self.all_attr(lambda emb: emb.dist_idx(idx))
        if self.riemann:
            return torch.norm(torch.stack(d, 0), 2, dim=0)
        else:
            return sum(d)
    def dist_row(self, i):
        # return sum([H.dist_row(i) for H in self.H])
        d = self.all_attr(lambda emb: emb.dist_row(i))
        if self.riemann:
            return torch.norm(torch.stack(d, 0), 2, dim=0)
        else:
            return sum(d)
    def dist_matrix(self):
        # return sum([H.dist_matrix() for H in self.H])
        d = self.all_attr(lambda emb: emb.dist_matrix())
        if self.riemann:
            return torch.norm(torch.stack(d), 2, dim=0)
        else:
            return sum(d)

    def loss(self, _x):
        idx, values, w = _x
        d = self.dist_idx(idx)

        #term_rescale = torch.exp( 2*(1.-values) ) if self.exponential_rescale else self.step_rescale(values)
        term_rescale = w

        if self.absolute_loss:
            loss = torch.sum( term_rescale*( d - values)**2)
        elif self.logrel_loss:
            loss = torch.sum( torch.log((d/values)**2)**2 )
        elif self.dist_loss:
            loss = torch.sum( torch.abs(term_rescale*((d/values) - 1)) )
        elif self.square_loss:
            loss = torch.sum( term_rescale*torch.abs((d/values)**2 - 1) )
        else:
            l1 = torch.sum( term_rescale*((d/values) - 1)**2 )
            l2 = torch.sum( term_rescale*((values/d) - 1)**2 ) if self.sym_loss else 0
            loss = l1 + l2
        return loss / values.size(0)

    def normalize(self):
        for H in self.H:
            H.normalize()
        for S in self.S:
            S.normalize()


class Embedding(nn.Module):
    def __init__(self, dist_fn, param_cls, n, d, project=True, initialize=None, learn_scale=False, initial_scale=0.0):
        super().__init__()
        self.dist_fn = dist_fn
        self.n, self.d = n, d
        self.project   = project
        if initialize is not None: logging.info(f"Initializing {np.any(np.isnan(initialize.numpy()))} {initialize.size()} {(n,d)}")
        # x      = h_proj( 1e-3 * torch.rand(n, d).double() ) if initialize is None  else torch.DoubleTensor(initialize[0:n,0:d])
        # self.w = Hyperbolic_Parameter(x)
        # self.w = param_cls(x)
        self.w = param_cls(data=initialize, sizes=(n,d))
        z =  torch.tensor([0.0], dtype=torch.double)
        # init_scale = 1.0
        if learn_scale:
            self.scale_log       = nn.Parameter(torch.tensor([initial_scale], dtype=torch.double))
            # self.scale_log.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
        else:
            self.scale_log       = torch.tensor([initial_scale], dtype=torch.double, device=device)
        self.learn_scale = learn_scale
        # self.scale_clamp       = 3.0
        # logging.info(f"{self} {torch.norm(self.w.data - x)} {x.size()}")
        logging.info(f"{self} {self.w.size()}")

    def scale(self):
        # print(self.scale_log.type(), self.lo_scale.type(), self.hi_scale.type())
        # scale = torch.exp(torch.clamp(self.scale_log, -self.thres, self.thres))
        # scale = torch.exp(self.scale_log.tanh()*self.scale_clamp)
        # return torch.sqrt(self.scale_log)
        scale = torch.exp(self.scale_log)
        # scale = self.scale_log
        # scale = scale if self.learn_scale else 1.0
        return scale

    def dist_idx(self, idx):
        # print("idx shape: ", idx.size(), "values shape: ", values.size())
        wi = torch.index_select(self.w, 0, idx[:,0])
        wj = torch.index_select(self.w, 0, idx[:,1])
        d = self.dist_fn(wi,wj)
        return d * self.scale() # rescale to the size of the true distances matrix

    def dist_row(self, i):
        m = self.w.size(0)
        return self.dist_fn(self.w[i,:].clone().unsqueeze(0).repeat(m,1), self.w) * self.scale()

    def dist_matrix(self):
        m    = self.w.size(0)
        rets = torch.zeros(m, m, dtype=torch.double, device=device)
        for i in range(m):
            rets[i,:] = self.dist_row(i)
        return rets

    def normalize(self):
        self.w.proj()
        # if self.project:
        #     self.w.proj()
            # print("normalize: scale ", self.scale.data)
            # print(type(self.scale_log), self.scale_log.type())
            # self.scale_log = torch.clamp(self.scale_log, self.lo_scale, self.hi_scale)
