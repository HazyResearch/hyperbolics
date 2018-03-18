import os
import argh
import numpy as np
import pandas
import networkx as nx
import scipy.sparse.csgraph as csg
from timeit import default_timer as timer

import utils.load_graph as lg
import distortions as dis
import graph_util as gu


def compute_row_stats(i, n, adj_mat_original, hyp_dist_row, weighted, verbose=False):
    print("i", i, "hyp_dist_row", hyp_dist_row)
    # the real distances in the graph
    # print(hyp_dist_row)
    true_dist_row = csg.dijkstra(adj_mat_original, indices=[i], unweighted=(not weighted), directed=False).squeeze()
    # print(true_dist_row)
    # print(true_dist_row.shape)
    # this is this row MAP
    # TODO: should this be n_bfs instead of n? n might include points that weren't embedded?
    curr_map  = dis.map_row(true_dist_row, hyp_dist_row, n, i)

    # these are distortions: worst cases (contraction, expansion) and average
    mc, me, avg, bad = dis.distortion_row(true_dist_row, hyp_dist_row, n, i)
    wc = mc*me

    # print out stats for this row
    if verbose:
        print(f"Row {i}, MAP = {curr_map}, distortion = {avg}, {wc}")

    return (curr_map, avg, wc)


@argh.arg("dataset", help="Dataset to compute stats for")
@argh.arg("d_file", help="File with embedded distance matrix")
@argh.arg("-v", "--verbose", help="Print more detailed stats")
def stats(dataset, d_file, verbose=False):
    start = timer()

    # Load graph
    G        = lg.load_graph(dataset)
    n = G.order()
    weighted = gu.is_weighted(G)

    adj_mat_original = nx.to_scipy_sparse_matrix(G, range(0,n))

    print(f"Finished loading graph. Elapsed time {timer()-start}")

    # Load distance matrix chunks
    # chunk_i = -1
    # maps = 0.0
    # d_avg = 0.0
    # wc = 0.0
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
    #     print(f"Finished loading distance matrix chunk. Elapsed time {toc-tic}")

    #     print(rows)
    #     _maps = np.zeros_like(rows)
    #     _d_avg = np.zeros_like(rows)
    #     _wc = np.zeros_like(rows)
    #     for (i, row) in enumerate(rows):
    #         print(i, row)
    #         (_maps[i], _d_avg[i], _wc[i]) = compute_row_stats(row, n, adj_mat_original, hyp_dist_mat[i,:], weighted=weighted, verbose=verbose)
    #         # print(_maps[i], _d_avg[i], _wc[i])
    #     maps += sum(_maps)
    #     d_avg += sum(_d_avg)
    #     wc = max(wc, np.max(_wc))

    #     tac = timer()
    #     print(f"Finished computing stats for chunk. Elapsed time {tac-toc}")

    hyp_dist_mat = pandas.read_csv(d_file, index_col=0).as_matrix()
    _maps = np.zeros(n)
    _d_avg = np.zeros(n)
    _wc = np.zeros(n)
    for i in range(n):
        (_maps[i], _d_avg[i], _wc[i]) = compute_row_stats(i, n, adj_mat_original, hyp_dist_mat[i,:], weighted=weighted, verbose=verbose)
        # print(_maps[i], _d_avg[i], _wc[i])
    maps = np.sum(_maps)
    d_avg = np.sum(_d_avg)
    wc = np.max(_wc)

    if weighted:
        print("Note: MAP is not well defined for weighted graphs")

    # Final stats:
    print(f"Final MAP = {maps/n}")
    print(f"Final d_avg = {d_avg/n}, d_wc = {wc}")

    end = timer()
    print(f"Finished computing stats. Total elapsed time {end-start}")

if __name__ == '__main__':
    _parser = argh.ArghParser() 
    # _parser.add_commands([build])
    # _parser.dispatch()
    _parser.set_default_command(stats)
    _parser.dispatch()
