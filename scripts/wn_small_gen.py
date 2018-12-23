import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import json
from collections import defaultdict
import matplotlib.pyplot as plt



def make_edge_set(): return ([],([],[]))
def add_edge(e,i,j):
    (v,(row,col)) = e
    row.append(i)
    col.append(j)
    v.append(1)



def get_hyponym_tree(syn,d,e,counter):
    if syn in d.keys():
        syn_idx = d[syn]
    elif syn not in d.keys():
        counter +=1
        syn_idx = counter
        d[syn] = syn_idx
    curr_list = syn.hyponyms()
    if len(curr_list) != 0:
        for hyp in curr_list:
            if hyp in d.keys():
                hyp_idx = d[hyp]
            elif hyp not in d.keys():
                counter +=1
                hyp_idx = counter
                d[hyp] = hyp_idx              
            add_edge(e, syn_idx, hyp_idx)
            add_edge(e, hyp_idx, syn_idx)
            e, d, counter = get_hyponym_tree(hyp,d,e,counter)
    return e, d, counter

d = dict()
IDtoWord = dict()
e = make_edge_set()
word = "attribute"
immediate_synsets = wn.synsets(word)
print(immediate_synsets)
counter = 0

num_syns = 0
for syn in immediate_synsets:
    num_syns += len(syn.hyponyms())
    d[syn] = 0
    e, d, counter = get_hyponym_tree(syn, d, e, counter)

#Get some stats.
mat_shape = max(e[1][1])
M = csr_matrix(e, shape=(mat_shape+1, mat_shape+1))
G = nx.from_scipy_sparse_matrix(M)
print("Number of edges:")
print(len(e[0]))
print("Number of nodes:")
print(max(e[1][1]))
print("Degree of main node:")
print(G.degree(0))
print("Degree should be:")
print(num_syns)

for key, val in d.items():
    if val != 0:
        name = key.name().split('.')[0]
        IDtoWord[val] = name
IDtoWord[0] = word


#Save stuff.

nx.write_edgelist(G, "data/edges/wn_small.edges",data=False)
json.dump(IDtoWord, open("data/edges/wn_small_dict.txt","w"))



# nx.draw_networkx(G, with_labels=True)
# plt.show()

