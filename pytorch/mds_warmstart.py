import utils.data_prep as dp
import pytorch.graph_helpers as gh
import numpy as np
import utils.distortions as dis
import utils.load_graph as load_graph
import torch, logging
from math import sqrt

def cudaify(x):            return x.cuda() if torch.cuda.is_available() else x

def compute_d(u,l,n):
    if np.min(u) < 0.:
        print(np.min(u))
        print(u)
        assert(False)

    b       =max(1. + (sum(u)/sqrt(l)*np.linalg.norm(u))**2,1.)

    alpha = b - np.sqrt(b**2-1.)
    v   = u*(l*(1.-alpha))/sum(u)
    d   = (v+1.)/(1.+alpha)
    d_min = np.min(d)
    if d_min < 1:
        print("\t\t\t Warning: Noisy d_min correction used.")
        #d/=d_min
    dv  = d - 1
    dinv = 1./d
    t     = dinv*np.divide(v-alpha,(1)+alpha)
    return (d,dv,t)

def data_rec(points, scale=1.0):
    (n,d) = points.shape
    Z     = np.zeros( (n,n) )
    for i in range(n):
        di = 1-np.linalg.norm(points[i,:])**2
        for j in range(n):
            dj = 1-np.linalg.norm(points[j,:])**2
            Z[i,j] = np.linalg.norm(points[i,:] - points[j,:])**2/(di*dj)
    return (Z,np.arccosh(1+2.*Z)/scale)

def center_numpy_inplace(tZ,inv_d,v):
    n = tZ.shape[0]
    for i in range(n):
        for j in range(n):
            tZ[i,j] *= inv_d[i]

    for i in range(n):
        for j in range(n):
            tZ[i,j] *= inv_d[j]

    for i in range(n):
        for j in range(n):
            tZ[i,j] -= (v[i]+v[j])

    #mu = np.mean(tZ,1)
    #for i in range(n): tZ[:,i] -= mu

    #mu = np.mean(tZ,0)
    #for i in range(n): tZ[i,:] -= mu

def power_method(_A,r,T=5000,tol=1e-14):
    (n,n) = _A.shape
    A     = cudaify( torch.DoubleTensor(_A) )
    x     = cudaify( torch.DoubleTensor( np.random.randn(n,r)/r ) )
    _eig  = cudaify( torch.DoubleTensor(r).zero_() )
    for i in range(r):
        for j in range(T):
            y      = x[:,0:i]@(x[:,0:i].transpose(0,1)@x[:,i]) if i > 0 else 0.
            x[:,i] = A@x[:,i] - y
            nx     = torch.norm(x[:,i])
            x[:,i] /= nx
            if (abs(_eig[i]) > tol) and (abs(_eig[i] - nx)/_eig[i] < tol):
                print(f"\teig {i} iteration {j} --> {_eig[i]}")
                break
            _eig[i] = nx

    return (_eig.cpu().numpy(), x.cpu().numpy())

def get_eig(A,r, use_power=False):
    if use_power:
        return power_method(A,r)
    else:
        e, ev = np.linalg.eigh(A)
        return e[-r:], ev[:,-r:]

def get_model(dataset, max_k, scale = 1.0):
    #G = dp.load_graph(dataset)
    G  = load_graph.load_graph(dataset)

    H = gh.build_distance(G,1.0)
    (n,n) = H.shape
    Z = (np.cosh(scale*H) -1)/2

    # Find Perron vector
    (d,U)=get_eig(Z,1)
    idx  = np.argmax(d)
    l0   = d[idx]
    u    = U[:,idx]
    u    = u if u[0] > 0 else -u

    (d1,dv,v) = compute_d(u,l0,n)
    inv_d = 1./d1
    #Q  = (np.eye(n)-np.ones( (n,n)) /n)*np.diag(inv_d)
    #G  = -Q@Z@Q.T/2
    G   = Z # This does make a copy.
    center_numpy_inplace(G, inv_d, v)
    G /= -2.0

    # Recover our points
    (emb_d, points_d) = get_eig(G,max_k)
    # good_idx = emb_d > 0
    # our_points = np.real(points_d[:,good_idx]@np.diag(np.sqrt(emb_d[good_idx])))
    bad_idx = emb_d <= 0
    emb_d[bad_idx] = 0
    our_points = points_d@np.diag(np.sqrt(emb_d))

    # Just for evaluation
    (Z,Hrec) = data_rec(our_points, scale)
    # np.set_printoptions(threshold=np.nan)
    # print(f"Distortion Score {dis.distortion(H, Hrec, n, 2)}")
    # this will get done in the preliminary stats pass:
    #print(f"Map Score {dis.map_score(H/scale, Hrec, n, 2)}")
    return (H,our_points)

def get_normalized_hyperbolic(model):
    x   = torch.DoubleTensor(model)
    ds  = torch.norm(x,2,1)
    ds2 = ds**2
    # need to find y s.t. \|x\|^2 = \frac{\|y\|^2}{1-\|y\|^2} => \|y\|^2 = \frac{\|x\|^2}{1+\|x\|^2}
    new_norm = torch.sqrt(ds2/((1+1e-10)*(1.0+ds2)))/ds
    z = torch.diag(new_norm) @ x
    logging.info(f"norm_z={torch.max(torch.norm(z,2,1))} min_norm_ds={torch.min(ds)} input={torch.max(torch.norm(x,2,1))} q={np.any(np.isnan(z.numpy()))}")
    return z
