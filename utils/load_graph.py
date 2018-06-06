# This is to load data
# the graph needs to be prepared; for example utils.data_prep preprocesses and saves prepared edge lists
import networkx as nx

# def load_graph(file_name, directed=False):
#     container = nx.DiGraph() if directed else nx.Graph()
#     G  = nx.read_edgelist(file_name, data=(('weight',float),), create_using=container)
#     G_comp = nx.convert_node_labels_to_integers(G)
#     return G_comp

def load_graph(file_name, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    with open(file_name, "r") as f:
        for line in f:
            tokens = line.split()
            u = int(tokens[0])
            v = int(tokens[1])
            if len(tokens) > 2:
                w = float(tokens[2])
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(u,v)
    return G

