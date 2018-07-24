# Baselines using ancestor encoding:
import networkx as nx
import os, sys
import subprocess

edges_dir = '../data/edges/'
all_files = os.listdir(edges_dir)

out = open('./spanning_forest_avgs.txt', 'w')

for file in all_files:
    if os.path.isdir(edges_dir+file):
        continue

    print("Working on ", edges_dir+file)
    G = nx.read_edgelist(edges_dir+file, data=False)

    # get the forest:
    G_comps = nx.connected_component_subgraphs(G)
    n_comps = 0

    avg_dists = []

    for comp in G_comps:
        n_comps += 1
        comp = nx.convert_node_labels_to_integers(comp)
        comp_bfs = nx.bfs_tree(comp, 0)

        dists = nx.shortest_path_length(comp_bfs, 0)
        tot_dists = sum(dists.values())
        avg_dist  = tot_dists/comp_bfs.order()

        avg_dists.append(avg_dist)

    # that's it for this graph:
    out.write(file + " ")
    out.write(str(sum(avg_dists)/n_comps) + "\n")
    out.write(str(avg_dists) + "\n")

out.close()