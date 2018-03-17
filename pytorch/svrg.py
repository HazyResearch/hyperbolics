# Copy and modified from
#
# https://github.com/mleszczy/pytorch_optimizers
#
from torch.optim.optimizer import Optimizer, required
import torch
import copy, logging 
from torch.autograd import Variable
from hyperbolic_parameter import Hyperbolic_Parameter


#TODO(mleszczy): Be able to inherit from different optimizers
# NB: Note we choose the baseclass dynamically below.
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

    def __init__(self, params, lr=required, T=required, data_loader=required, weight_decay=0.0,opt=torch.optim.SGD):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        self.__class__ = type(self.__class__.__name__,
                              (opt,object),
                              dict(self.__class__.__dict__))
        logging.info(f"Using base optimizer {opt} in SVRG")
        super(self.__class__, self).__init__(params, **defaults)

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
        self._full_grad = None
        
        self.data_loader = data_loader

        if T == 0:
            T = len(self.data_loader)*3
        logging.info(f"SVRG epoch: {T} batches")
            
        self.state['t_iters'] = T

        self._first_call = True
        self.T = T # Needed to trigger full gradient
        logging.info(f"Data Loader has {len(self.data_loader)} with batch {self.data_loader.batch_size}")

    def __setstate__(self, state):
        super(self.__class__, self).__setstate__(state)
                
    def _zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()

    def _set_weights_grad(self,ws,gs):
        for idx, p in enumerate(self._params):
            if ws is not None: p.data = ws[idx]
            if gs is not None and p.grad is not None: p.grad.data = gs[idx]

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Calculate full gradient
        if self.state['t_iters'] == self.T or self._first_call:      
            # Reset gradients before accumulating them 
            self._set_weights_grad(None, self._full_grad)
            self._zero_grad()
                    
            # Accumulate gradients
            for i, (data, target) in enumerate(self.data_loader):
                closure(data, target)
                
            # Adjust summed gradients by num_iterations accumulated over
            # assert(n_iterations == len(self.data_loader))
            for p in self._params:
                if p.grad is not None:
                    p.grad.data /= len(self.data_loader)

            # As the gradient is lazily allocated, on the first call,
            # we make a copy.
            if self._first_call:
                assert(self._full_grad is None)
                self._full_grad = [p.grad.data.clone() if p.grad is not None else None for p in self._params]
                self._first_call = False
                
            # Copy w to prev_w
            for p, p0 in zip(self._curr_w, self._prev_w):
                p0.copy_(p)

            # Reset t 
            self.state['t_iters'] = 0
        
        # Setup the previous grad
        self._set_weights_grad(self._prev_w, self._prev_grad)        
        self._zero_grad()
        closure()

        # Calculate the current grad.
        self._set_weights_grad(self._curr_w, self._curr_grad)
        self._zero_grad()
        loss = closure()

        # Adjust the current gradient using the previous gradient and the full gradient.
        # We have normalized so that these are all comparable.
        for p, d_p0, fg in zip(self._params, self._prev_grad, self._full_grad):
            # Adjust gradient in place
            if p.grad is not None:
                # NB: This should be _this_ batch.
                p.grad.data -= (d_p0 - fg) 

        # Call optimizer update step
        # TODO: Abstract this away.
        Hyperbolic_Parameter.correct_metric(self._params)
        super(self.__class__, self).step()
        
        self.state['t_iters'] += 1 
        return loss
