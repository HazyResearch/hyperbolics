import logging, argh
import os, sys
import networkx as nx
import random

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import matplotlib as mpl
if torch.cuda.is_available(): mpl.use('Agg')
import matplotlib.pyplot as plt
if torch.cuda.is_available(): plt.ioff()
import scipy
import scipy.sparse.csgraph as csg
import pandas
import numpy as np, math

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
# import utils.load_graph as load_graph
# import utils.vis as vis
# import utils.distortions as dis
# import graph_helpers as gh
# import mds_warmstart
from hyperbolic_models import ProductEmbedding
import json
from hyperbolic_parameter import RParameter



with open("wn_IDtoSyns.txt") as d:
    IDtoSyns = json.load(d)

m = torch.load("wordnet_full.emb")

spherical_embs = [S.w for S in m.S]
euclidean_embs = [E.w for E in m.E]
hyperbolic_embs = [H.w for H in m.H]

hyperbolic_matrix = (hyperbolic_embs[0].cpu()).data.numpy()
scale = np.float64(m.scale_params[0].cpu().data.numpy())

print(hyperbolic_matrix)
print(scale)

#Matching the IDs to entities.

final_emb = dict()
for i in range(0, hyperbolic_matrix.shape[0]):
    syn = IDtoSyns[str(i)]
    vector = hyperbolic_matrix[i]
    final_emb[syn] = vector

lines = ["Scaling factor "+str(scale)]
for key in final_emb.keys():
    curr_line = str(key) + " " + " ".join(list(map(str,final_emb[key])))
    lines.append(curr_line)


with open('wordnet_full.txt', 'w') as f:
    f.write('\n'.join(lines))





