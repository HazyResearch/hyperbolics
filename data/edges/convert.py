import networkx as nx
import utils.data_prep as dp

def convert_to_edges(opt, name):
    G = dp.load_graph(opt)
    nx.write_edgelist(G, "data/edges/" + name + ".edges", data=False)
