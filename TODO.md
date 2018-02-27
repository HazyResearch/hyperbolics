# hyperbolics
Hyperbolic Embeddings 


### TODO
0. Documentation and data cleanup.

1. Pytorch Code Cleanup
   * Unify warm-start code
     * Tree-based warm start
     * Fix scaling for hMDS warm-start
   * Get rid of "number based" datasets and allow file loads.
   * ~~SVRG to accelerate convergence~~.
   * ~~Multithreading to load the sampler~~
   
2. (h)MDS.
   * Low precision. Speed improvements using HazyTensor
   * High precision. ~~Multithreading!~~
   * Add Scale to the saved representaiton for hMDS

3. Combinatorial
   * Refactor to use arbitrary data sets.
   * Refactor to support warm start.

4. Datasets.
   * Run wiki articles.
   * Clean-up run scripts for all experiments to make reproduction easier.
     * Snapshot logs for runs.
