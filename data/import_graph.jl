using PyCall
@pyimport networkx as nx
@pyimport scipy.sparse.csgraph as csg

G = nx.read_edgelist("grqc.edgelist")
C = nx.to_scipy_sparse_matrix(G)