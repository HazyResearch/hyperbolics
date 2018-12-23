import torch
from torch import nn
from torch.autograd import Variable
import logging
import numpy as np, math
import random

def dot(x,y): return torch.sum(x * y, -1)
def acosh(x):
    return torch.log(x + torch.sqrt(x**2-1))


class RParameter(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None, exp=False):
        if data is None:
            assert sizes is not None
            data = (1e-3 * torch.randn(sizes, dtype=torch.double)).clamp_(min=-3e-3,max=3e-3)
        #TODO get partial data if too big i.e. data[0:n,0:d]
        ret =  super().__new__(cls, data, requires_grad=requires_grad)
        # ret.data    = data
        ret.initial_proj()
        ret.use_exp = exp
        return ret

    @staticmethod
    def _proj(x):
        raise NotImplemented

    def proj(self):
        self.data = self.__class__._proj(self.data.detach())
        # print(torch.norm(self.data, dim=-1))

    def initial_proj(self):
        """ Project the initialization of the embedding onto the manifold """
        self.proj()

    def modify_grad_inplace(self):
        pass

    @staticmethod
    def correct_metric(ps):
        for p in ps:
            if isinstance(p,RParameter):
                p.modify_grad_inplace()


# TODO can use kwargs instead of pasting defaults
class HyperboloidParameter(RParameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None, exp=True):
        if sizes is not None:
            sizes = list(sizes)
            sizes[-1] += 1
        return super().__new__(cls, data, requires_grad, sizes, exp)

    @staticmethod
    def dot_h(x,y):
        return torch.sum(x * y, -1) - 2*x[...,0]*y[...,0]
    @staticmethod
    def norm_h(x):
        assert torch.all(HyperboloidParameter.dot_h(x,x) >= 0), torch.min(HyperboloidParameter.dot_h(x,x))
        return torch.sqrt(torch.clamp(HyperboloidParameter.dot_h(x,x), min=0.0))
    @staticmethod
    def dist_h(x,y):
        # print("before", x, y)
        # print("before dots", HyperboloidParameter.dot_h(x,x)+1, HyperboloidParameter.dot_h(y,y)+1)
        # print("after dots", -HyperboloidParameter.dot_h(x,y))
        # return acosh(-HyperboloidParameter.dot_h(x,y) - 1e-7)
        bad = torch.min(-HyperboloidParameter.dot_h(x,y) - 1.0)
        if bad <= -1e-4:
            print("bad dist", bad.item())
        # assert torch.all(-HyperboloidParameter.dot_h(x,y) >= 1.0 - 1e-4), torch.min(-HyperboloidParameter.dot_h(x,y) - 1.0)
	    # we're dividing by dist_h somewhere so we can't have it be 0, force dp > 1
        return acosh(torch.clamp(-HyperboloidParameter.dot_h(x,y), min=(1.0+1e-8)))

    @staticmethod
    def _proj(x):
        """ Project onto hyperboloid """
        x_ = torch.tensor(x)
        x_tail = x_[...,1:]
        current_norms = torch.norm(x_tail,2,-1)
        scale      = (current_norms/1e7).clamp_(min=1.0)
        x_tail /= scale.unsqueeze(-1)
        x_[...,1:] = x_tail
        x_[...,0] = torch.sqrt(1 + torch.norm(x_tail,2,-1)**2)

        debug = True
        if debug:
            bad = torch.min(-HyperboloidParameter.dot_h(x_,x_))
            if bad <= 0.0:
                print("way off hyperboloid", bad)
            assert torch.all(-HyperboloidParameter.dot_h(x_,x_) > 0.0), f"way off hyperboloid {torch.min(-HyperboloidParameter.dot_h(x_,x_))}"
        xxx = x_ / torch.sqrt(torch.clamp(-HyperboloidParameter.dot_h(x_,x_), min=0.0)).unsqueeze(-1)
        return xxx
        # return x / (-HyperboloidParameter.norm_h(x)).unsqueeze(-1)

    def initial_proj(self):
        """ Project the initialization of the embedding onto the manifold """
        self.data[...,0] = torch.sqrt(1 + torch.norm(self.data.detach()[...,1:],2,-1)**2)
        self.proj()


    def exp(self, lr):
        """ Exponential map """
        x = self.data.detach()
        # print("norm", HyperboloidParameter.norm_h(x))
        v = -lr * self.grad

        retract = False
        if retract:
        # retraction
            # print("retract")
            self.data = x + v

        else:
            # print("tangent", HyperboloidParameter.dot_h(x, v))
            assert torch.all(1 - torch.isnan(v))
            n = self.__class__.norm_h(v).unsqueeze(-1)
            assert torch.all(1 - torch.isnan(n))
            n.clamp_(max=1.0)
            # e = torch.cosh(n)*x + torch.sinh(n)*v/n
            mask = torch.abs(n)<1e-7
            cosh = torch.cosh(n)
            cosh[mask] = 1.0
            sinh = torch.sinh(n)
            sinh[mask] = 0.0
            n[mask] = 1.0
            e = cosh*x + sinh/n*v
            # assert torch.all(-HyperboloidParameter.dot_h(e,e) >= 0), torch.min(-HyperboloidParameter.dot_h(e,e))
            self.data = e
        self.proj()


    def modify_grad_inplace(self):
        """ Convert Euclidean gradient into Riemannian """
        self.grad[...,0] *= -1
        #print("check data")
        #print(np.argwhere(torch.isnan(self.data).cpu().numpy()))
        #print("check grad")
        #print(np.argwhere(torch.isnan(self.grad).cpu().numpy()))


        # self.grad += self.__class__.dot_h(self.data, self.grad).unsqueeze(-1) * self.data
        self.grad -= self.__class__.dot_h(self.data, self.grad).unsqueeze(-1) / HyperboloidParameter.dot_h(self.data, self.data).unsqueeze(-1) * self.data


# TODO:
# 1. Improve speed up of projection by making operations in place.
class PoincareParameter(RParameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None, check_graph=False):
        ret =  super().__new__(cls, data, requires_grad, sizes)
        ret.check_graph = check_graph
        return ret

    def modify_grad_inplace(self):
        # d        = self.data.dim()
        w_norm   = torch.norm(self.data,2,-1, True)
        # This is the inverse of the remanian metric, which we need to correct for.
        hyper_b  = (1 - w_norm**2)**2/4
        # new_size = tuple([1] * (d - 1) + [self.data.size(d-1)])
        # self.grad   *= hyper_b.repeat(*new_size) # multiply pointwise
        self.grad   *= hyper_b # multiply pointwise
        self.grad.clamp_(min=-10000.0, max=10000.0)

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
        return modified.unsqueeze(-1)

    @staticmethod
    def _proj(x, eps=1e-10):
        return x * PoincareParameter._correct(x, eps=eps)

    # def proj(self, eps=1e-10):
    #     self.data = self.__class__._proj(self.data.detach())#PoincareParameter._correct(self.data, eps=eps)

    def __repr__(self):
        return 'Hyperbolic parameter containing:' + self.data.__repr__()

class SphericalParameter(RParameter):
    def __new__(cls, data=None, requires_grad=True, sizes=None, exp=True):
        if sizes is not None:
            sizes = list(sizes)
            sizes[-1] += 1
        return super().__new__(cls, data, requires_grad, sizes, exp)


    def modify_grad_inplace(self):
        """ Convert Euclidean gradient into Riemannian by projecting onto tangent space """
        # pass
        self.grad -= dot(self.data, self.grad).unsqueeze(-1) * self.data

    def exp(self, lr):
        x = self.data.detach()
        v = -lr*self.grad

        retract = False
        if retract:
        # retraction
            self.data = x + v

        else:
            n = torch.norm(v, 2, -1, keepdim=True)
            mask = torch.abs(n)<1e-7
            cos = torch.cos(n)
            cos[mask] = 1.0
            sin = torch.sin(n)
            sin[mask] = 0.0
            n[torch.abs(n)<1e-7] = 1.0
            e = cos*x + sin*v/n
            self.data = e
        self.proj()

    @staticmethod
    def _proj(x):
        # return x / torch.norm(x, 2, -1).unsqueeze(-1)
        return x / torch.norm(x, 2, -1, True)

    # def proj(self):
    #     x = self.data.detach()
    #     self.data = SphericalParameter._proj(x)
    def initial_proj(self):
        # pass
        self.data[...,0] = torch.sqrt(1 - torch.norm(self.data[...,1:],2,-1)**2)

class EuclideanParameter(RParameter):
    def proj(x):
        pass

