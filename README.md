# hyperbolics
Hyperbolic embedding implementations of [Representation Tradeoffs for Hyperbolic Embeddings](https://arxiv.org/pdf/1804.03329.pdf).

<p align="center">
  <img src="assets/binary_tree.png" alt="Hyperbolic embedding of binary tree" width="400"/>
</p>

## Setup
We use Docker to set up the environment for our code. See Docker/README.md for installation and launch instructions.

In this README, all instructions are assumed to be run inside the Docker container. All paths are relative to the /hyperbolics directory, and all commands are expected to be run from this directory.


## Usage
The following programs and scripts expect the input graphs to exist in the /data/edges folder, e.g. /data/edges/phylo_tree.edges. All graphs that we report results on have been prepared and saved here.


### Combinatorial construction
`julia combinatorial/comb.jl --help` to see options. Example usage (for better results on this dataset, raise the precision):

```
julia combinatorial/comb.jl -d data/edges/phylo_tree.edges -m phylo_tree.r10.emb -e 1.0 -p 64 -r 10 -a -s
```

### Pytorch optimizer
`python pytorch/pytorch_hyperbolic.py learn --help` to see options. Optimizer requires torch >=0.4.1. Example usage:

```
python pytorch/pytorch_hyperbolic.py learn data/edges/phylo_tree.edges --batch-size 64 --dim 10 -l 5.0 --epochs 100 --checkpoint-freq 10 --subsample 16
```

Products of hyperbolic spaces with Euclidean and spherical spaces are also supported. E.g. adding flags `-euc 1 -edim 20 -sph 2 -sdim 10` embeds into a product of Euclidean space of dimension 20 with two copies of spherical space of dimension 10.

### Experiment scripts
* `scripts/run_exps.py` runs a full set of experiments for a list of datasets. Example usage (note: the default run settings take a long time to finish):
    ```
    python scripts/run_exps.py phylo -d phylo_tree --epochs 20
    ```

    Currently, it executes the following experiments:
    1. The combinatorial construction with fixed precision in varying dimensions
    2. The combinatorial construction in dimension 2 (Sarkar's algorithm), with very high precision
    3. Pytorch optimizer in varying dimensions, random initialization
    4. Pytorch optimizer in varying dimensions, using the embedding produced by the combinatorial construction as initialization 

* The combinatorial constructor `combinatorial/comb.jl` has an option for reporting the MAP and distortion statistics. However, this can be slow on larger datasets such as wordnet
    * `scripts/comb_stats.py` provides an alternate method for computing stats that can leverage multiprocessing
        Example usage: `python scripts/comb_stats.py phylo_tree -e 1.0 -r 2 -p 1024 -q 4` to run on 4 cores

<!--

[comment]: # ( scripts/comb_stats.py for embedding and stats just for combinatorial construction)

[comment]: # (    * this is intended specifically for computing statistics for the combinatorial embedding on large datasets. for other uses, e.g. generating the embedding for downstream use, it is recommended to use the basic program)

[comment]: # (    * will save temporary files to distances/ directory)

[comment]: # (    * If the dataset is large (wordnet), you will see stats for every batch and aggregate statistics at the end)

[comment]: # (        * warning about overloading files; if you play with batch size in this code, you might need to clear this directory after every run)

-->
