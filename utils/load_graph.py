# This is to load data
import networkx as nx
import scipy as sp
import numpy as np

def load_graph(file_name):
    G  = nx.read_edgelist(file_name, data=(('weight',float),)) 
    
    # work with integers as the node names    
    G = nx.convert_node_labels_to_integers(G)
    
    # take the largest component
    G_comp_unsort = max(nx.connected_component_subgraphs(G), key=len)

    # the connected_component function changes the edge orders, so fix:
    G_comp_sorted = nx.Graph()
    G_comp_sorted.add_edges_from(sorted(G_comp_unsort.edges(data=True)))
    G_comp = nx.convert_node_labels_to_integers(G_comp_sorted)
    
    return G_comp
