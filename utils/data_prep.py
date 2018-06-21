# This is to load all of our data
import networkx as nx
import scipy as sp
import numpy as np

from Bio import Phylo
# import nltk.corpus as nc
# import utils.word_net_prep as wnp

def load_graph(opt):
    if opt == 1:
        G = nx.read_edgelist("data/facebook_combined.txt")
    elif opt == 2:
        G = nx.read_edgelist("data/cithepph.txt")
    elif opt == 3:
        G = nx.read_edgelist("data/grqc.edgelist")
    elif opt == 4:
        G = nx.read_edgelist("data/wikilinks.tsv")
    elif opt == 5:
        G = nx.read_edgelist("data/california.edgelist")
    elif opt == 6:
        tree = Phylo.read("data/T92308.nex", "nexus")
        G = Phylo.to_networkx(tree)
        G = nx.convert_node_labels_to_integers(G)
        G = G.to_undirected()
    elif opt == 7:
        G = nx.read_edgelist("data/bio-diseasome.mtx")
    elif opt == 8:
        G = nx.read_edgelist("data/bio-yeast.mtx")
    elif opt == 9:
        G = nx.read_edgelist("data/inf-power.mtx")
    elif opt == 10:
        G = nx.read_edgelist("data/web-edu.mtx")
    elif opt == 11:
        G = nx.read_edgelist("data/ca-CSphd.mtx")
    elif opt == 12:
        G = nx.balanced_tree(3,3)
    elif opt == 13:
        G = nx.balanced_tree(2,2)
    elif opt == 14:
        (n,C) = wnp.load_big_component()
        G = nx.Graph(C).to_undirected();
    else:
        assert(False)
    # take the largest component
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    G_comp_unsort = max(nx.connected_component_subgraphs(Gc), key=len)

    # the connected_component function changes the edge orders, so fix:
    G_comp_sorted = nx.Graph()
    G_comp_sorted.add_edges_from(sorted(G_comp_unsort.edges()))
    G_comp = nx.convert_node_labels_to_integers(G_comp_sorted)

    return G_comp

def save_edges(G, name, data=False):
    if data:
        nx.write_weighted_edgelist(G, "data/edges/" + name + ".edges")
    else:
        nx.write_edgelist(G, "data/edges/" + name + ".edges", data)

def make_wordnet_weights():
    (n,C) = wnp.load_big_component()
    G = nx.Graph(C).to_undirected()
    Gc = max(nx.connected_component_subgraphs(G), key=len)

    # 'entity' is 0:
    G_BFS = nx.bfs_tree(Gc, 0)
    G_W   = nx.Graph()

    # each edge must be appropriately weighted:
    curr_nodes = [0]
    next_nodes = []
    depth = 0

    while 1:
        if len(curr_nodes) == 0:
            if len(next_nodes) == 0:
                break
            depth += 1
            curr_nodes = next_nodes.copy()
            next_nodes.clear()

        node = curr_nodes[0]
        parent = list(G_BFS.predecessors(node))
        if len(parent) > 0:
            G_W.add_edge(node, parent[0], weight=2**(depth-1))

        curr_nodes.remove(node)
        next_nodes += list(G_BFS.successors(node))

    save_edges(G_W, "weighted_wordnet", data=True)
