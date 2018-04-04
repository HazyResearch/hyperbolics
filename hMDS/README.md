# h-MDS
Julia scripts to perform h-MDS.

usage: hmds-simple.jl -d DATASET -r DIM -t SCALE [-m SAVE-EMBEDDING]
                     [-q PROCS] [-h]  

  -d, --dataset DATASET  
                        Dataset to embed  
  -r, --dim DIM         Dimension r  
  -t, --scale SCALE     Scaling factor  
  -m, --save-embedding  Save embedding to file  
  -q, --procs PROCS     Number of processes to use  
  -h, --help            show this help message and exit  

Example on small attached tree. 

```
julia hMDS/hmds-simple.jl -d data/edges/phylo_tree.edges -r 100 -t 0.1 -m savetest.csv  
```

Output:  
h-MDS. Info:  
Data set = data/edges/phylo_tree.edges  
Dimensions = 100  
Save embedding to savetest  
Scaling = 0.1  

Number of nodes is  344  
Time elapsed =  0.17120695114135742  
Doing h-MDS...  
elapsed time: 2.771179635 seconds  
elapsed time: 4.263825574 seconds  
Building recovered graph...  
elapsed time: 0.297506376 seconds  
Getting metrics...  

Distortion avg, max, bad = 0.032613823778721615, 4.335859234069171, 762.0  
MAP = 0.6170225406053894  
