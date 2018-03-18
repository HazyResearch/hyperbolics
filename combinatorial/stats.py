import os
import argh
import numpy as np
import pandas
import networkx as nx
import scipy.sparse.csgraph as csg
from timeit import default_timer as timer
from multiprocessing import Pool

import utils.load_graph as lg
import distortions as dis
import graph_util as gu


def compute_row_stats(i, n, adj_mat_original, hyp_dist_row, weighted, verbose=False):
    # print("i", i, "hyp_dist_row", hyp_dist_row)
    # the real distances in the graph
    # print(hyp_dist_row)
    true_dist_row = csg.dijkstra(adj_mat_original, indices=[i], unweighted=(not weighted), directed=False).squeeze()
    # print(true_dist_row)
    # print(true_dist_row.shape)
    # this is this row MAP
    # TODO: should this be n_bfs instead of n? n might include points that weren't embedded?
    curr_map  = dis.map_row(true_dist_row, hyp_dist_row, n, i)

    # these are distortions: worst cases (contraction, expansion) and average
    dc, de, avg, bad = dis.distortion_row(true_dist_row, hyp_dist_row, n, i)
    wc = dc*de

    # print out stats for this row
    if verbose:
        print(f"Row {i}, MAP = {curr_map}, distortion = {avg}, d_c = {dc}, d_e = {de}")

    return (curr_map, avg, dc, de)


@argh.arg("dataset", help="Dataset to compute stats for")
@argh.arg("d_file", help="File with embedded distance matrix")
# @argh.arg("-s", "--save", help="File to save final stats to")
@argh.arg("-q", "--procs", help="Number of processors to use")
@argh.arg("-v", "--verbose", help="Print more detailed stats")
def stats(dataset, d_file, procs=1, verbose=False):
    start = timer()

    # Load graph
    G        = lg.load_graph(dataset)
    n = G.order()
    weighted = gu.is_weighted(G)

    adj_mat_original = nx.to_scipy_sparse_matrix(G, range(0,n))

    print(f"Finished loading graph. Elapsed time {timer()-start}")

    # Load distance matrix chunks
    # chunk_i = -1
    # n_ = 0
    # map_ = 0.0
    # d_avg_ = 0.0
    # wc_ = 0.0
    # while True:
    #     tic = timer()
    #     chunk_i += 1
    #     chunk_file = f"{d_file}.{chunk_i}"
    #     chunk_exists = os.path.isfile(chunk_file)
    #     if not chunk_exists:
    #         break
    #     hyp_dist_df = pandas.read_csv(chunk_file, index_col=0)
    #     rows = hyp_dist_df.index.values
    #     hyp_dist_mat = hyp_dist_df.as_matrix()

    #     toc = timer()
    #     print(f"Finished loading distance matrix chunk {chunk_i}. Elapsed time {toc-tic}")

    #     _map = np.zeros(rows.shape)
    #     _d_avg = np.zeros(rows.shape)
    #     _wc = np.zeros(rows.shape)

    #     for (i, row) in enumerate(rows):
    #         (_map[i], _d_avg[i], _wc[i]) = compute_row_stats(row, n, adj_mat_original, hyp_dist_mat[i,:], weighted=weighted, verbose=verbose)
    #     # with Pool(processes=proc) as pool:

    #     n_ += rows.size
    #     map_ += np.sum(_map)
    #     d_avg_ += np.sum(_d_avg)
    #     wc_ = max(wc_, np.max(_wc))

    #     # Running stats:
    #     print(f"Running MAP = {map_/n_}")
    #     print(f"Running d_avg = {d_avg_/n_}, d_wc = {wc_}")

    #     tac = timer()
    #     print(f"Finished computing stats for chunk. Elapsed time {tac-toc}")

    #hyp_dist_mat = pandas.read_csv(d_file, index_col=0).as_matrix()
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
    # for i in range(n):
        # (_map[i], _d_avg[i], _wc[i]) = compute_row_stats(i, n, adj_mat_original, hyp_dist_mat[i,:], weighted=weighted, verbose=verbose)
    for (i, row) in enumerate(rows):
        (_map[i], _d_avg[i], _dc[i], _de[i]) = compute_row_stats(row, n, adj_mat_original, hyp_dist_mat[i,:], weighted=weighted, verbose=verbose)
    map_ = np.sum(_map)
    d_avg_ = np.sum(_d_avg)
    dc_ = np.max(_dc)
    de_ = np.max(_de)

    if weighted:
        print("Note: MAP is not well defined for weighted graphs")

    # Final stats:
    print(f"MAP = {map_/n_}")
    print(f"d_avg = {d_avg_/n_}, d_c = {dc_}, d_e = {de_}")

    end = timer()
    print(f"Finished computing stats. Total elapsed time {end-start}")

    with open(f"{d_file}.stats", "w") as stats_log:
        stats_log.write(f"{n_},{map_},{d_avg_},{dc_},{de_}\n")

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    # _parser.add_commands([build])
    # _parser.dispatch()
    _parser.set_default_command(stats)
    _parser.dispatch()
