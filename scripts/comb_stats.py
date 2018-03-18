import sys, os
import subprocess

if __name__ == '__main__':
    # print(sys.argv)
    dataset = sys.argv[1]
    flags = sys.argv[2:]

    os.makedirs("distances", exist_ok=True)

    edge_file = f"data/edges/{dataset}.edges"
    distance_file = f"distances/{dataset}{''.join(flags)}.dist"


    comb_cmd = ['julia', 'combinatorial/comb.jl',
                    '--dataset', edge_file,
                    '--save-distances', distance_file] + flags
    print(comb_cmd)
    subprocess.run(comb_cmd)

    stats_cmd = ['python', 'combinatorial/stats.py', edge_file, distance_file]
    print(stats_cmd)
    subprocess.run(stats_cmd)
    # TODO: pipe result here, compile stats into a dataframe somehow

    # julia combinatorial/comb.jl -d data/edges/ca-CSphd.edges -e 1.0 -r 2 -p 16384 -y ca-CSphd.dist
    # python combinatorial/stats.py data/edges/phylo_tree.edges phylo_tree.dist
