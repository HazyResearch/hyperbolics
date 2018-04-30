
import nltk
from nltk.corpus import wordnet as wn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import floyd_warshall, connected_components
from collections import defaultdict
import numpy as np
import networkx as nx
import json
import time
from collections import defaultdict


def make_edge_set(): return ([],([],[]))
def add_edge(e, i,j):
    (v,(row,col)) = e
    row.append(i)
    col.append(j)
    v.append(1)

def add_big_edge(e, i,j):
    (v,(row,col)) = e
    row.append(i)
    col.append(j)
    v.append(100)


def load_wordnet():
    d        = dict()
    ID_dict = dict()

    all_syns = list(wn.all_synsets())

    for idx, x in enumerate(all_syns): 
        d[x] = idx
        ID_dict[idx] = x.name().split('.')[0]

    n = len(all_syns)
    e = make_edge_set()

    for idx, x in enumerate(all_syns):
        for y in x.hypernyms():
            y_idx = d[y]
            add_edge(e, idx  , y_idx)
            add_edge(e, y_idx,   idx)

    return e, d, ID_dict, all_syns, csr_matrix(e,shape=(n, n))


def load_connected_components():

    e, d, ID_dict, all_syns, X = load_wordnet()
    C = connected_components(X)


    mat_shape = len(C[1])
    prev_comp_idx = 0

    print("There are "+str(C[0])+ " connected components.")

    for num in range(C[0]):
        begin = time.time()
        curr_comp = np.array(all_syns)[C[1] == num]
        print(curr_comp)
        print(len(curr_comp))
        # mat_shape += len(curr_comp)

        curr_comp_idx = d[curr_comp[0]]
        if num!=0:
            add_big_edge(e, prev_comp_idx , curr_comp_idx)
            add_big_edge(e, curr_comp_idx, prev_comp_idx)
        prev_comp_idx = curr_comp_idx
        print(str(num)+"th cc took "+str(time.time()-begin))



    wordID_dict = defaultdict(list)
    for key in d.keys():
        for word in key.lemma_names():
            if "_" not in word:
                idx = d[key]
                wordID_dict[word].append(idx)

    X2  = csr_matrix(e, shape=(mat_shape, mat_shape))
    return (ID_dict, wordID_dict, d, mat_shape, X2)


if __name__ == '__main__':
    ID_dict, wordID_dict, d, n, G = load_connected_components()
    edges = nx.from_scipy_sparse_matrix(G)
    print("writing to the file")
    nx.write_weighted_edgelist(edges, "embeddings/wordnet_all.edges")
    json.dump(ID_dict,open("embeddings/IDstoWords.txt","w"))
    json.dump(wordID_dict,open("embeddings/WordstoIDs.txt","w"))

    with open('embeddings/WordstoIDs.txt', 'r') as inf2:
        WordstoIDs = eval(inf2.read())

    with open('embeddings/wordnet100.emb', 'r') as emb:
        emb_lines = emb.readlines()

    emb_lines = emb_lines[1:]
    vector_dict = dict()
    for idx, line in enumerate(emb_lines):
        curr_line = line.split(',')[:-1]
        vector_dict[int(curr_line[0])] = np.asarray(list(map(np.float64, curr_line[1:])))

    #Create the dictionary for final embedding (does simple Euclidean averaging)
    final_emb = dict()
    for word in WordstoIDs.keys():
        counter = 0
        curr_sum = np.zeros(vector_dict[0].shape)
        for idx in WordstoIDs[word]:
            curr_sum += vector_dict[idx]
            counter +=1
        final_emb[word] = curr_sum/counter

    lines = []
    for key in final_emb.keys():
        curr_line = str(key) + " " + " ".join(list(map(str,final_emb[key])))
        lines.append(curr_line)

    with open('embeddings/wordnet.100d.txt', 'w') as f:
        f.write('\n'.join(lines))




