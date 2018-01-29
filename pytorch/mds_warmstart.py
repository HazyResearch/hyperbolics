import data_prep as dp
import pytorch.graph_helpers as gh
import numpy as np
import distortions as dis
def compute_d(u,l,n):
    assert( np.min(u) >= 0. )
    b       = 1. + sum(u)**2/(l*np.linalg.norm(u)**2)
    
    alpha = b - np.sqrt(b**2-1.)
    v   = u*(l*(1.-alpha))/sum(u)
    d   = (v+1.)/(1.+alpha)
    d_min = np.min(d)
    if d_min < 1:
        print("\t\t\t Warning: Noisy d_min correction used.")
        d/=d_min
    dv  = d - 1 
    return (d,dv)

def data_rec(points, scale=1.0):
    (n,d) = points.shape
    Z     = np.zeros( (n,n) )
    for i in range(n):
        #di = 1-np.linalg.norm(points[i,:])**2
        for j in range(n):
            #dj = 1-np.linalg.norm(points[j,:])**2
            Z[i,j] = np.linalg.norm(points[i,:] - points[j,:])**2#/(di*dj)
    return (Z,np.arccosh(1+2.*Z)/scale)

def get_model(dataset, scale = 1.0):
    G = dp.load_graph(dataset)
    H = gh.build_distance(G,1.0)
    (n,n) = H.shape
    Z = (np.cosh(scale*H) -1)/2

    # Find Perron vector
    (d,U)=np.linalg.eig(Z)
    idx  = np.argmax(d)
    l0   = d[idx]
    u    = U[:,idx]
    u    = u if u[0] > 0 else -u

    (d1,dv) = compute_d(u,l0,n)
    inv_d = 1./d1
    Q  = (np.eye(n)-np.ones( (n,n)) /n)*np.diag(inv_d)
    G  = -Q@Z@Q.T/2

    # Recover our points
    (emb_d, points_d) = np.linalg.eig(G)
    good_idx = emb_d > 0
    our_points = points_d[:,good_idx]@np.diag(np.sqrt(emb_d[good_idx]))

    # Just for evaluation
    (Z,Hrec) = data_rec(our_points)
    print(f"Map Score {dis.map_score(H/scale, Hrec, n, 2)}")
    return (H,our_points)
