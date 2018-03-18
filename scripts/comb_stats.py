import sys, os, subprocess
import shutil
import numpy as np
import pandas

def comb(edge_file, distance_file, flags):
    comb_cmd = ['julia', 'combinatorial/comb.jl',
                    '--dataset', edge_file,
                    '--save-distances', distance_file] + flags
    print(comb_cmd)
    subprocess.run(comb_cmd)

def stats(edge_file, distance_file):
    # stats_cmd = ['python', 'combinatorial/stats.py', edge_file, distance_file]
    # print(stats_cmd)
    # subprocess.run(stats_cmd)

    # Load distance matrix chunks
    chunk_i = -1
    n_ = 0
    map_ = 0.0
    d_avg_ = 0.0
    wc_ = 0.0
    files = []
    while True:
        chunk_i += 1
        chunk_file = f"{distance_file}.{chunk_i}"
        chunk_exists = os.path.isfile(chunk_file)
        if not chunk_exists:
            break
        files.append(chunk_file)

    parallel_cmd = ['parallel', 'python', 'combinatorial/stats.py', edge_file, ':::'] + files
    print(parallel_cmd)
    subprocess.run(parallel_cmd)

    stats_file = f"{distance_file}.stats"
    cat_cmd = ['cat'] + [f+'.stats' for f in files]
    with open(stats_file, "w") as s:
        subprocess.run(cat_cmd, stdout=s)

    _stats = pandas.read_csv(stats_file, header=None, index_col=False).as_matrix()
    n_ = np.sum(_stats[:,0])
    map_ = np.sum(_stats[:,1])
    d_avg_ = np.sum(_stats[:,2])
    wc_ = np.max(_stats[:,3])

    # print(_stats)
    print(f"Final MAP = {map_/n_}")
    print(f"Final d_avg = {d_avg_/n_}, d_wc = {wc_}")



if __name__ == '__main__':
    # print(sys.argv)
    dataset = sys.argv[1]
    flags = sys.argv[2:]

    # TODO: with the chunking, this currently fails if there was a previous run with a smaller chunk size (i.e. there are preexisting files that do not get overwritten by the new run)
    os.makedirs(f"distances/{dataset}", exist_ok=True)

    edge_file = f"data/edges/{dataset}.edges"
    distance_file = f"distances/{dataset}/{dataset}{''.join(flags)}.dist"

    # comb(edge_file, distance_file, flags)
    stats(edge_file, distance_file)

    # TODO: pipe result here, compile stats into a dataframe somehow

    # julia combinatorial/comb.jl -d data/edges/ca-CSphd.edges -e 1.0 -r 2 -p 16384 -y ca-CSphd.dist
    # python combinatorial/stats.py data/edges/phylo_tree.edges phylo_tree.dist
