# Copy and modified from
#
# https://github.com/mleszczy/pytorch_optimizers
#
from torch.optim.optimizer import Optimizer, required
import torch
import copy 
from torch.autograd import Variable
from hyperbolic_parameter import Hyperbolic_Parameter

def set_weights_grad(ps,ws,gs):
    for idx, p in enumerate(ps):
        if ws is not None: p.data = ws[idx]
        if gs is not None and p.grad is not None: p.data = gs[idx]
        
#TODO(mleszczy): Be able to inherit from different optimizers 
class SVRG(torch.optim.SGD):
    r"""Implements stochastic variance reduction gradient descent.
    Args:
        params (iterable): iterable of parameters to optimize 
        lr (float): learning rate
        T (int): number of iterations between the step to take the full grad/save w
        data_loader (DataLoader): dataloader to use to load training data
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
    .. note::
    """

    def __init__(self, params, lr=required, T=required, data_loader=required, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SVRG, self).__init__(params, **defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SVRG doesn't support per-parameter options "
                             "(parameter groups)")

        # TODO(mleszczy): Add these to parameter group or state?
        params = self.param_groups[0]['params']
        
        self._params = params

        self._curr_w = [p.data for p in params]
        self._prev_w = [p.data.clone() for p in params] 

        # Gradients are lazily allocated and don't exist yet. However, gradients are 
        # the same shape as the weights so we can still allocate buffers here
        self._curr_grad = [p.data.clone() for p in params]
        self._prev_grad = [p.data.clone() for p in params]
        self._full_grad = [p.data.clone() for p in params]
        
        self.data_loader = data_loader
        self.state['t_iters'] = T
        self.T = T

    def __setstate__(self, state):
        super(SVRG, self).__setstate__(state)

    # This is only changing the pointer to data and not copying data 
    def _switch_weights_to_copy(self, copy_w):
        for (w_new, p) in zip(copy_w, self._params):
            p.data = w_new

    # This is actually copying data (setting pointers to grad.data didn't work)
    def _copy_grads_from_params(self, grad_buffer):
        for idx, (grad_data, p) in enumerate(zip(grad_buffer, self._params)):
            if p.grad is not None:
                grad_data.copy_(p.grad.data)
                
    def _zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Calculate full gradient 
        if self.state['t_iters'] == self.T:  
            # Setup the full grad
            # for p,_p in zip(self._params,self._full_grad):
            #     if p.grad is not None:
            #         p.grad.data = _p
            set_weights_grad(self._params,None, self._full_grad)
            
            # Reset gradients before accumulating them 
            self._zero_grad()
                    
            # Accumulate gradients
            for i, (data, target) in enumerate(self.data_loader):
                closure(data, target)
            
            # Adjust summed gradients by num_iterations accumulated over 
            for p in self._params:
                if p.grad is not None:
                    p.grad.data /= len(self.data_loader)

            # self._copy_grads_from_params(self._full_grad)
                
            # Copy w to prev_w
            for p, p0 in zip(self._curr_w, self._prev_w):
                p0.copy_(p)

            # Reset t 
            self.state['t_iters'] = 0
            # Restore the pointers
            # for p,_p in zip(self._params,self._curr_grad):
            #     if p.grad is not None:
            #         p.grad.data = _p
            #        assert(p.grad.data.data_ptr() == _p.data_ptr())
            set_weights_grad(self._params, None, self._curr_grad)
        # Copy prev_w over to model parameters
        #self._switch_weights_to_copy(self._prev_w)
        # for p,_w,_g in zip(self._params,self._prev_w, self._prev_grad):
        #     p.data = _w
        #     if p.grad is not None:
        #         p.grad.data = _g
        set_weights_grad(self._params,self._prev_w, self._prev_grad)        
        self._zero_grad()
        
        # Calculate prev_w gradient 
        closure()
        #self._copy_grads_from_params(self._prev_grad)

        # Copy w over to model parameters
        #self._switch_weights_to_copy(self._curr_w)
        # for p,_w,_g in zip(self._params,self._curr_w, self._curr_grad):
        #     p.data = _w
        #     if p.grad is not None:
        #        p.grad.data = _g

        set_weights_grad(self._params,self._curr_w, self._curr_grad)
        self._zero_grad()
        # Calculate w gradient 
        loss = closure()
        # We don't need to copy out these gradients

        for p, d_p0, fg in zip(self._params, self._prev_grad, self._full_grad):
            # Adjust gradient in place
            if p.grad is not None:
                p.grad.data -= (d_p0 - fg) 

        # Call optimizer update step
        Hyperbolic_Parameter.correct_metric(self._params)
        super(SVRG, self).step()
       
      
        self.state['t_iters'] += 1 

        # TODO(mleszczy): What to return -- is this loss value useful? 
        return loss
