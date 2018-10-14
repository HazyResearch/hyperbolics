import nltk
from nltk.corpus import wordnet as wn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import floyd_warshall, connected_components
import operator
from collections import defaultdict
import numpy as np
import networkx as nx
import json
from collections import defaultdict

"""Script assumes input text is in GloVe format."""



# Some definitions

# Reflection (circle inversion of x through orthogonal circle centered at a)
def isometric_transform(a, x):
    r2 = np.linalg.norm(a)**2 - (1.0)
    return r2/np.linalg.norm(x - a)**2 * (x-a) + a

# Inversion taking mu to origin
def reflect_at_zero(mu,x):
    a = mu/np.linalg.norm(mu)**2
    return isometric_transform(a,x)

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
    sgn = (-1.0)**float(w<0)
    w *= sgn    
    if w == 1:
        return sgn*x
    else:
        x_dist = (1+np.linalg.norm(x))/(1-np.linalg.norm(x))
        alpha = 1-2/(1+x_dist**w)
        alpha *= 1/np.linalg.norm(x)
    
    return sgn*alpha*x

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


vector_dim = 21
file = "wordnet_full.txt"


with open(file, 'r') as emb:
    emb_lines = emb.readlines()
    relTau = np.float64(emb_lines[0])
    emb_lines = emb_lines[1:]
    emb_dict = dict()
    IDtoWords = dict()
    WordstoIDs = dict()
    for idx, line in enumerate(emb_lines):
        curr_line = line.split(" ")
        curr_syn = curr_line[0]                
        emb_dict[curr_syn] = np.asarray(list(map(np.float64, curr_line[1:])))
        IDtoWords[idx] = curr_syn
        WordstoIDs[curr_syn] = idx


    
vocab_size = len(emb_dict)
W = np.zeros((vocab_size, vector_dim))
for word, vec in emb_dict.items():
    idx = WordstoIDs[word]
    W[idx,:] = vec



# Find the top 10 nearest neighbors to a particular synset for given relationship.

e1 = wn.synset('geometry.n.01')
vec_e1 = emb_dict[str(e1)]
curr_dist = []  
for row_idx in range(W.shape[0]):
    curr_vec = W[row_idx,:]
    normalized_dist = (dist(curr_vec,vec_e1))/relTau
    curr_dist.append(normalized_dist)

e1_idx = WordstoIDs[str(e1)]
curr_dist[e1_idx] = np.Inf
curr_closest_indices = np.argsort(curr_dist)[:10]
for r_idx in curr_closest_indices:
    relev_syn = IDtoWords[r_idx]
    print(relev_syn)


# Analogy experiments.
e1 = wn.synset('plane_geometry.n.01')
e1_idx = WordstoIDs[str(e1)]

e2 = wn.synset('geometry.n.01')
e2_idx = WordstoIDs[str(e2)]

e3 = wn.synset('novelist.n.01')
e3_idx = WordstoIDs[str(e3)]


vec_e1 = emb_dict[str(e1)]
vec_e2 = emb_dict[str(e2)]
vec_e3 = emb_dict[str(e3)]

vec1_ = hyp_scale(-1, vec_e1)
left_sum = hyp_weighted_sum(1, 1, vec_e2, vec1_)
vec_search = hyp_weighted_sum(1, 1, left_sum, vec_e3)

curr_dist = []    
for row_idx in range(W.shape[0]):
    curr_vec = W[row_idx,:]
    normalized_dist = (dist(curr_vec, vec_search))/relTau
    curr_dist.append(normalized_dist)

curr_dist[e1_idx] = np.Inf
curr_dist[e2_idx] = np.Inf
curr_dist[e3_idx] = np.Inf

curr_closest_indices = np.argsort(curr_dist)[:10]
for r_idx in curr_closest_indices:
    relev_syn = IDtoWords[r_idx]
    print(relev_syn)

