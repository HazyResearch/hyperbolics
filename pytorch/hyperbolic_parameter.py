import torch
from torch import nn
from torch.autograd import Variable
import logging
import numpy as np, math
import random

class RParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None):
        if data is None:
            assert sizes is not None
            data = 1e-3 * torch.randn(sizes, dtype=torch.double)
        #TODO get partial data if too big i.e. data[0:n,0:d]
        ret =  super().__new__(cls, data, requires_grad=requires_grad)
        # ret.data    = data
        ret.proj()
        return ret

    @staticmethod
    def _proj(x):
        raise NotImplemented

    def proj(self):
        self.data = self.__class__._proj(self.data)


# TODO:
# 1. Improve speed up of projection by making operations in place.
class HyperbolicParameter(RParameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None, check_graph=False):
        ret =  super().__new__(cls, data, requires_grad, sizes)
        ret.check_graph = check_graph
        return ret

    # def __init__(self, data=None, require_grad=True, sizes=None, check_graph=False):
    #     super().__init__()
    #     if x is not None:
    #         self.data    = data
    #     else:
    #         assert sizes is not None
    #         self.data = 1e-3 * torch.randn(sizes, dtype=torch.double)
    #     self.proj()
    #     self.check_graph = check_graph

    def modify_grad_inplace(self):
        # d        = self.data.dim()
        w_norm   = torch.norm(self.data,2,-1, True)
        # This is the inverse of the remanian metric, which we need to correct for.
        hyper_b  = (1 - w_norm**2)**2/4
        # new_size = tuple([1] * (d - 1) + [self.data.size(d-1)])
        # self.grad.data   *= hyper_b.repeat(*new_size) # multiply pointwise
        self.grad.data   *= hyper_b # multiply pointwise
        self.grad.data.clamp_(min=-10000.0, max=10000.0)

        # We could do the projection here?
        # NB: THIS IS DEATHLY SLOW. FIX IT
        if self.check_graph and np.any(np.isnan(self.grad.data.cpu().numpy())):
             print(np.any(np.isnan(self.data.cpu().numpy())))
             print(np.any(np.isnan(self.grad.data.cpu().numpy())))
             print(np.any(np.isnan(w_norm.cpu().numpy())))
             raise ValueError("NaN During Hyperbolic")

    @staticmethod
    def _correct(x, eps=1e-10):
        current_norms = torch.norm(x,2,x.dim() - 1)
        mask_idx      = current_norms < 1./(1+eps)
        modified      = 1./((1+eps)*current_norms)
        modified[mask_idx] = 1.0
        #new_size      = [1]*current_norms.dim() + [x.size(x.dim()-1)]
        #return modified.unsqueeze(modified.dim()).repeat(*new_size)
        # return modified.unsqueeze(modified.dim()).expand(x.size())
        return modified.unsqueeze(modified.dim())

    @staticmethod
    def _proj(x, eps=1e-10):
        return x * HyperbolicParameter._correct(x, eps=eps)

    # def proj(self, eps=1e-10):
    #     self.data = self.__class__._proj(self.data.detach())#HyperbolicParameter._correct(self.data, eps=eps)

    @staticmethod
    def correct_metric(ps):
        for p in ps:
            if isinstance(p,HyperbolicParameter):
                p.modify_grad_inplace()

    def __repr__(self):
        return 'Hyperbolic parameter containing:' + self.data.__repr__()

class SphericalParameter(RParameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None):
        if sizes is not None:
            sizes = list(sizes)
            sizes[-1] += 1
        return super().__new__(cls, data, requires_grad, sizes)

    # def __init__(self, data=None, require_grad=True, sizes=None):
    #     super().__init__()
    #     if data is not None:
    #         self.data    = data 
    #     else:
    #         assert sizes is not None
    #         self.data = 1e-3 * torch.randn(sizes, dtype=torch.double)
    #     self.proj()

    @staticmethod
    def _proj(x):
        # return x / torch.norm(x, 2, -1).unsqueeze(-1)
        return x / torch.norm(x, 2, -1, True)

    # def proj(self):
    #     x = self.data.detach()
    #     self.data = SphericalParameter._proj(x)

class EuclideanParameter(RParameter):
    @staticmethod
    def _proj(x):
        pass
    def proj(x):
        pass

