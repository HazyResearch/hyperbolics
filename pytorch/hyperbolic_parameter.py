import torch
from torch import nn
from torch.autograd import Variable
import logging
import numpy as np, math
import random

# TODO:
# 1. Improve speed up of projection by making operations in place.
class Hyperbolic_Parameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, project=True, check_graph=False):
        ret =  super().__new__(cls, data, requires_grad=requires_grad)
        if project: ret.proj()
        ret.project = project
        ret.data    = data
        ret.check_graph = check_graph
        return ret

    def __init__(self, x):
        super().__init__()
        self.data    = x

    def modify_grad_inplace(self):
        d        = self.data.dim()
        w_norm   = torch.norm(self.data,2,d - 1, True)
        # This is the inverse of the remanian metric, which we need to correct for.
        hyper_b  = (1 - w_norm**2)**2/4
        new_size = tuple([1] * (d - 1) + [self.data.size(d-1)])

        self.grad.data   *= hyper_b.repeat(*new_size) # multiply pointwise
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
        mask_idx      = current_norms < 1.0
        modified      = 1./((1+eps)*current_norms)
        modified[mask_idx] = 1.0
        #new_size      = [1]*current_norms.dim() + [x.size(x.dim()-1)]
        #return modified.unsqueeze(modified.dim()).repeat(*new_size)
        return modified.unsqueeze(modified.dim()).expand(x.size())

    @staticmethod
    def _proj(x, eps=1e-10):
        return x * Hyperbolic_Parameter._correct(x, eps=eps)

    def proj(self, eps=1e-10):
        self.data *= Hyperbolic_Parameter._correct(self.data, eps=eps)

    @staticmethod
    def correct_metric(ps):
        for p in ps:
            if isinstance(p,Hyperbolic_Parameter):
                p.modify_grad_inplace()

    def __repr__(self):
        return 'Hyperbolic parameter containing:' + self.data.__repr__()
