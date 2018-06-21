import os
import argh
import numpy as np
import pandas
import networkx as nx
import scipy.sparse.csgraph as csg
from timeit import default_timer as timer
from multiprocessing import Pool

import utils.load_graph as lg
import utils.distortions as dis
import graph_util as gu


def compute_row_stats(i, n, adj_mat_original, hyp_dist_row, weighted, verbose=False):
    # the real distances in the graph
    true_dist_row = csg.dijkstra(adj_mat_original, indices=[i], unweighted=(not weighted), directed=False).squeeze()
    # true_dist_row = csg.dijkstra(adj_mat_original, indices=[i], unweighted=True, directed=True).squeeze()
    # print(f"{i}: {true_dist_row}")

    # row MAP
    neighbors = adj_mat_original.todense()[i].A1
    # print(f"row {i}: ", neighbors)
    # print("shape", neighbors.shape)
    row_map  = dis.map_row(neighbors, hyp_dist_row, n, i)

    # distortions: worst cases (contraction, expansion) and average
    dc, de, avg, _ = dis.distortion_row(true_dist_row, hyp_dist_row, n, i)
    # dc, de, avg = 0.0, 0.0, 0.0

    # print out stats for this row
    if verbose:
        print(f"Row {i}, MAP = {curr_map}, distortion = {avg}, d_c = {dc}, d_e = {de}")

    return (row_map, avg, dc, de)


@argh.arg("dataset", help="Dataset to compute stats for")
@argh.arg("d_file", help="File with embedded distance matrix")
# @argh.arg("-s", "--save", help="File to save final stats to")
@argh.arg("-q", "--procs", help="Number of processors to use")
@argh.arg("-v", "--verbose", help="Print more detailed stats")
def stats(dataset, d_file, procs=1, verbose=False):
    start = timer()

    # Load graph
    G        = lg.load_graph(dataset, directed=True)
    n = G.order()
    weighted = gu.is_weighted(G)
    print("G: ", G.edges)

    adj_mat_original = nx.to_scipy_sparse_matrix(G, range(0,n))

    print(f"Finished loading graph. Elapsed time {timer()-start}")

    # Load distance matrix chunks

    hyp_dist_df = pandas.read_csv(d_file, index_col=0)
    loaded = timer()
    print(f"Finished loading distance matrix. Elapsed time {loaded-start}")
    rows = hyp_dist_df.index.values
    hyp_dist_mat = hyp_dist_df.as_matrix()
    n_ = rows.size

    _map = np.zeros(n_)
    _d_avg = np.zeros(n_)
    _dc = np.zeros(n_)
    _de = np.zeros(n_)
    for (i, row) in enumerate(rows):
        # if row == 0: continue
        (_map[i], _d_avg[i], _dc[i], _de[i]) = compute_row_stats(row, n, adj_mat_original, hyp_dist_mat[i,:], weighted=weighted, verbose=verbose)
    map_ = np.sum(_map)
    d_avg_ = np.sum(_d_avg)
    dc_ = np.max(_dc)
    de_ = np.max(_de)

    if weighted:
        print("Note: MAP is not well defined for weighted graphs")

    # Final stats:
    # n_ -= 1
    print(f"MAP = {map_/n_}, d_avg = {d_avg_/n_}, d_wc = {dc_*de_}, d_c = {dc_}, d_e = {de_}")

    end = timer()
    print(f"Finished computing stats. Total elapsed time {end-start}")

    with open(f"{d_file}.stats", "w") as stats_log:
        stats_log.write(f"{n_},{map_},{d_avg_},{dc_},{de_}\n")
        print(f"Stats saved to {d_file}.stats")
    print()

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    _parser.set_default_command(stats)
    _parser.dispatch()
