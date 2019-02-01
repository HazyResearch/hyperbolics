"""This file contains core hyperbolic operations for learning modules."""

import numpy as np
import random
import os
import logging
from numpy import linalg as la
from numpy import random

import torch
import torch.nn.functional as F
import torch.nn as nn

EPS = 1e-15
PROJ_EPS = 1e-5
MAX_TANH_ARG = 15.0


def torch_norm(x):
    return torch.norm(x, dim=1, keepdim=True)

def torch_project_hyp_vec(v, c=1):
    """Projects the hyperbolic vectors to the inside of the ball."""
    # clip_norm = torch.tensor(1-PROJ_EPS)
    # clipped_v = F.normalize(v, p=2, dim=1)*clip_norm   
    # return clipped_v
    return v

def t_arctanh(v):
    return 0.5*torch.log((1+v)/(1-v))

def torch_lambda_x(x, c=1):
    return 2. / (1 - c * torch.dot(x,x))

def torch_dot(x, y):
    return torch.sum(x * y, dim=1, keepdim=True)

def torch_atanh(x):
    return t_arctanh(torch.min(x, torch.tensor(1. - EPS)))

def torch_tanh(x):
    return torch.tanh(torch.min(torch.max(x, torch.tensor(-MAX_TANH_ARG)), torch.tensor(MAX_TANH_ARG)))

def torch_lambda_x(x, c=1):
    return 2./(1-torch_dot(x,x))

def hyp_add_mob(u, v, c=1):
    num = (1.0 + 2.0 * c * np.dot(u, v) + c * la.norm(v)**2) * \
        u + (1.0 - c * la.norm(u)**2) * v
    denom = 1.0 + 2.0 * c * np.dot(u, v) + c**2 * la.norm(v)**2 * la.norm(u)**2
    return num/denom


def torch_hyp_add(u, v, c=1):
    """Accepts torch tensors u, v and returns their sum in hyperbolic
    space in tensor format. Radius of the open ball is 1/sqrt(c). """

    v = v+torch.tensor(EPS)
    torch_dot_u_v = 2 * torch_dot(u, v)
    torch_norm_u_sq = torch_dot(u,u)
    torch_norm_v_sq = torch_dot(v,v)
    denominator = 1. + torch_dot_u_v + torch_norm_v_sq * torch_norm_u_sq
    result = (1. + torch_dot_u_v + torch_norm_v_sq) / denominator * u + (1. - torch_norm_u_sq) / denominator * v
    return torch_project_hyp_vec(result)

#This is our definition which is compatible.
def hyp_scale_amb(r, x):
    """Scales x in hyperbolic space with r using the ambient space approach."""

    if r == 1:
        return x
    else:
        x_dist = (1+np.linalg.norm(x))/(1-np.linalg.norm(x))
        alpha = 1-2/(1+x_dist**r)
        alpha *= 1/np.linalg.norm(x)
        product = alpha*x

    return product

def hyp_scale_exp(r, x):
    """Scalar mult using exp map approach."""

    return exp_map(0, r*log_map(0, x))

def hyp_add(u, v, c=1):
    num = (1.0 + 2.0 * c * np.dot(u, v) + c * la.norm(v)**2) * \
        u + (1.0 - c * la.norm(u)**2) * v
    denom = 1.0 + 2.0 * c * np.dot(u, v) + c**2 * la.norm(v)**2 * la.norm(u)**2
    return num/denom


def exp_map(x, v, c=1):
    term = np.tanh(np.sqrt(c) * 2. / (1 - c * la.norm(x)**2) *
                   la.norm(v) / 2) / (np.sqrt(c) * la.norm(v)) * v
                   
    return hyp_add_mob(x, term, c)


def torch_scale_exp(r, x):
    """Scalar mult using exp map approach in torch."""

    zero = torch.zeros(x.shape)
    return torch_exp_map(zero, r*torch_log_map(zero, x))

def log_map(x, y, c=1):
    diff = hyp_add_mob(-x, y, c)
    lam = 2. / (1 - c * la.norm(x)**2)
    return 2. / (np.sqrt(c) * lam) * np.arctanh(np.sqrt(c) * la.norm(diff)) / (la.norm(diff)) * diff

def torch_exp_map(x, v, c=1):
    """Exp map for the vector v lying on the tangent space T_xM to 
    the point x in the manifold M."""
    v = v + torch.tensor(EPS)
    norm_v = torch_norm(v)
    term = (torch_tanh(torch_lambda_x(x, c) * norm_v / 2) / (norm_v)) * v
    return torch_hyp_add(x, term, c)

def torch_log_map_x(x, y, c=1):
    diff = torch_hyp_add(-x, y, c)+torch.tensor(EPS)
    norm_diff = torch_norm(diff)
    lam = torch_lambda_x(x, c)
    return ( (2 / lam) * torch_atanh(norm_diff) / norm_diff) * diff

def torch_exp_map_zero(v, c=1):
    # v = v + EPS # Perturbe v to avoid dealing with v = 0
    v=v+torch.tensor(EPS)
    norm_v = torch_norm(v)
    result = torch_tanh(norm_v) / (norm_v) * v
    return torch_project_hyp_vec(result, c)

def torch_log_map_zero(y, c=1):
    # diff = y + EPS
    diff = y+torch.tensor(EPS)
    norm_diff = torch_norm(diff)
    return torch_atanh(norm_diff) / norm_diff * diff


def mv_mul_hyp(M, x, c=1):
    Mx_norm = la.norm(M.dot(x))
    x_norm = la.norm(x)
    return 1. / np.sqrt(c) * np.tanh(Mx_norm / x_norm * np.arctanh(np.sqrt(c) * x_norm)) / Mx_norm * (M.dot(x))


def torch_mv_mul_hyp(M, x, c=1):
    x = x + torch.tensor(EPS)
    Mx = torch.matmul(x, M)+torch.tensor(EPS)
    MX_norm = torch_norm(Mx)
    x_norm = torch_norm(x)
    result = torch_tanh(MX_norm / x_norm * torch_atanh(x_norm)) / MX_norm * Mx
    return torch_project_hyp_vec(result, c)

# x is hyperbolic, u is Euclidean. Computes diag(u) \otimes x.
def torch_pointwise_prod(x, u, c=1):
    x = x+torch.tensor(EPS)
    Mx = x * u + torch.tensor(EPS)
    MX_norm = torch_norm(Mx)
    x_norm = torch_norm(x)
    result = torch_tanh(MX_norm / x_norm * torch_atanh(x_norm)) / MX_norm * Mx
    return torch_project_hyp_vec(result, c)


def hyp_non_lin(v, activation):
    logmap = log_map(0, v, 1)
    return exp_map(v, activation(logmap), 1)


def euclidean_softmax(x):
    """Euclidean softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def hyp_softmax_k(a_k, p_k, x, c=1):
    #This needs to be done for every class k in number of classes.
    """Hyperbolic softmax.
    a_k is a Euclidean and p_k is a hyperbolic parameter."""
    minus_p_plus_x = torch_hyp_add(-p_k, x, c)
    norm_a = torch.norm(a_k)
    lambda_px = torch_lambda_x(minus_p_plus_x, c)
    px_dot_a = torch.dot(minus_p_plus_x, a_k/norm_a)
    return 2 * norm_a * torch.asinh(px_dot_a * lambda_px)














