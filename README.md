# hyperbolics
Hyperbolic Embeddings 


## scripts
* load datasets

* basic commands: julia, pytorch
    * note PYTHONPATH

* scripts/comb_stats.py for embedding and stats just for combinatorial construction
    * this is intended specifically for computing statistics for the combinatorial embedding on large datasets. for other uses, e.g. generating the embedding for downstream use, it is recommended to use the basic program
    * will save temporary files to distances/ directory
    * If the dataset is large (wordnet), you will see stats for every batch and aggregate statistics at the end
        * warning about overloading files; if you play with batch size in this code, you might need to clear this directory after every run

* scripts/run.py for running full experiment set (intended for small datasets)


# TODOs 3/18
* add to Dockerfile: download parallel and suppress citation message


### TODO
0. Documentation and data cleanup.

1. Pytorch Code Cleanup
   * Tree-based warm start
   * Fix scaling for hMDS warm-start
   * Get rid of "number based" datasets.
   * SVRG to accelerate convergence.
   
2. (h)MDS.
   * Low precision. Speed improvements using HazyTensor
   * High precision. Multithreading!

3. Combinatorial
   * Refactor to use arbitrary data sets.
   * Refactor to support warm start.

4. Datasets.
   * Run wiki articles.

