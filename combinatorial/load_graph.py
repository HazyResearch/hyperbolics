# This is to load data
import networkx as nx
import scipy as sp
import numpy as np

def load_graph(file_name):
    G  = nx.read_edgelist(file_name)    
    
    # take the largest component
    G_comp = max(nx.connected_component_subgraphs(G), key=len)
    
    # Work with integers as the node names
    G_comp = nx.convert_node_labels_to_integers(G_comp)

    return G_comp
