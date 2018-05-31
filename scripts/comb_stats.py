import sys, os, subprocess
import shutil
import numpy as np
import pandas

def comb(edge_file, distance_file, flags):
    comb_cmd = ['julia', 'combinatorial/comb.jl',
                '--dataset', edge_file,
                '--save-distances', distance_file] + flags
    print(comb_cmd)
    print()
    subprocess.run(comb_cmd)

def stats(edge_file, distance_file):
    # Find distance matrix chunks
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

    parallel_stats_cmd = ['parallel', 'python', 'combinatorial/stats.py', edge_file, ':::'] + files
    print(parallel_stats_cmd)
    print()
    print("before subprocess call")
    subprocess.run(parallel_stats_cmd)

    stats_file = f"{distance_file}.stats"
    cat_cmd = ['cat'] + [f+'.stats' for f in files]
    with open(stats_file, "w") as s:
        subprocess.run(cat_cmd, stdout=s)

    _stats = pandas.read_csv(stats_file, header=None, index_col=False).as_matrix()
    n_ = np.sum(_stats[:,0])
    map_ = np.sum(_stats[:,1])
    d_avg_ = np.sum(_stats[:,2])
    dc_ = np.max(_stats[:,3])
    de_ = np.max(_stats[:,4])

    print(f"Final MAP = {map_/n_}")
    print(f"Final d_avg = {d_avg_/n_}, d_wc = {dc_*de_}, d_c = {dc_}, d_e = {de_}")



if __name__ == '__main__':
    dataset = sys.argv[1]
    stats_dataset = sys.argv[2]
    flags = sys.argv[3:]

    os.makedirs(f"distances/{dataset}", exist_ok=True)

    edge_file = f"data/edges/{dataset}.edges"
    stats_edge_file = f"data/edges/{stats_dataset}.edges"
    distance_file = f"distances/{dataset}/{dataset}{''.join(flags)}.dist"

    comb(edge_file, distance_file, flags)
    stats(stats_edge_file, distance_file)
