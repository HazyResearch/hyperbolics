# hyperbolics
Hyperbolic Embeddings 


## Setup

We use Docker to set up the environment for our code. See Docker/README.md for installation and launch instructions.

In this README, all instructions are assumed to be run inside the Docker container. All paths are relative to the /hyperbolics directory, and all commands are expected to be run from this directory.


## Usage
The following programs and scripts expect the input graphs to exist in the /data/edges folder, e.g. /data/edges/phylo_tree.edges. All graphs we report results on have been prepared and saved.


### Combinatorial construction
`julia combinatorial/comb.jl --help` to see options. Example usage:
    ``` julia combinatorial/comb.jl julia combinatorial/comb.jl -d data/edges/smalltree.edges -e 1.0 -p 256 -r 2 -s ```

### Pytorch optimizer
`pytorch pytorch/pytorch_hyperbolic.py learn --help` to see options. Example usage:
    ``` python pytorch/pytorch_hyperbolic.py learn data/edges/phylo_tree.edges --batch-size 64 -r 100 --epochs 500 --checkpoint-freq 100 -w combinatorial/phylo_tree.save ```

### Experiment scripts
* `scripts/run_exps.py` is a script that runs a full set of experiments for given datasets. Example usage: `python scripts/run_exps.py small -d smalltree`

    Currently, it executes the following experiments:
    1. The combinatorial construction with fixed precision in varying dimensions
    2. The combinatorial construction in dimension 2 (Sarkar's algorithm), with very high precision
    3. Pytorch optimizer in varying dimensions, random initialization
    4. Pytorch optimizer in varying dimensions, using the embedding produced by the combinatorial construction as initialization 

* The combinatorial constructor `combinatorial/comb.jl` has an option for reporting statistics such as MAP and distortion. However, this can be slow on larger datasets such as wordnet
    * `scripts/comb_stats.py` provides an alternate method for computing stats that uses multiprocessing


[//]: # (scripts/comb_stats.py for embedding and stats just for combinatorial construction)

[//]: # (    * this is intended specifically for computing statistics for the combinatorial embedding on large datasets. for other uses, e.g. generating the embedding for downstream use, it is recommended to use the basic program)

[//]: # (    * will save temporary files to distances/ directory)

[//]: # (    * If the dataset is large (wordnet), you will see stats for every batch and aggregate statistics at the end)

[//]: # (        * warning about overloading files; if you play with batch size in this code, you might need to clear this directory after every run)

