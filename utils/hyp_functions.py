# library of useful hyperbolic functions
import numpy as np

# Reflection (circle inversion of x through orthogonal circle centered at a)
def isometric_transform(a, x):
    r2 = np.linalg.norm(a)**2 - (1.0)
    return r2/np.linalg.norm(x - a)**2 * (x-a) + a

# Inversion taking mu to origin
def reflect_at_zero(mu,x):
    a = mu/np.linalg.norm(mu)**2
    return isometric_transform(a,x)

# Why isn't this in numpy?
def acosh(x):
    return np.log(x + np.sqrt(x**2-1))

# Hyperbolic distance
def dist(u,v):
    z  = 2 * np.linalg.norm(u-v)**2
    uu = 1. + z/((1-np.linalg.norm(u)**2)*(1-np.linalg.norm(v)**2))
    return acosh(uu)

# Hyperbolic distance from 0
def hyp_dist_origin(x):
    return np.log((1+np.linalg.norm(x))/(1-np.linalg.norm(x)))

# Scalar multiplication w*x
def hyp_scale(w, x):
    if w == 1:
        return x
    else:
        x_dist = (1+np.linalg.norm(x))/(1-np.linalg.norm(x))
        alpha = 1-2/(1+x_dist**w)
        alpha *= 1/np.linalg.norm(x)

    return alpha*x

# Convex combination (1-w)*x+w*y
def hyp_conv_comb(w, x, y):
    # circle inversion sending x to 0
    (xinv, yinv) = (reflect_at_zero(x, x), reflect_at_zero(x, y))
    # scale by w 
    pinv = hyp_scale(w, yinv)
    # reflect back
    return reflect_at_zero(x, pinv)

# Weighted sum w1*x + w2*y
def hyp_weighted_sum(w1, w2, x, y):
    p = hyp_conv_comb(w2 / (w1 + w2), x, y)
    return hyp_scale(w1 + w2, p)