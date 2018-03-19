# This is to load data
# the graph needs to be prepared; for example utils.data_prep preprocesses and saves prepared edge lists
import networkx as nx

def load_graph(file_name):
    G  = nx.read_edgelist(file_name, data=(('weight',float),))
    G_comp = nx.convert_node_labels_to_integers(G)
    return G_comp
