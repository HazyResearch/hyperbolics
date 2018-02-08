using PyCall
using JLD
#using GenericSVD
@pyimport numpy as np
@pyimport networkx as nx
@pyimport scipy.sparse.csgraph as csg
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport data_prep as dp
@pyimport load_dist as ld
@pyimport distortions as dis

data_set = parse(Int32,ARGS[1])
fname   = ARGS[2]
println("Loading")
tic()
X_f=load(fname)
Xrec = X_f["Xmds"]
toc()
(d,n) = size(Xrec)
println("Building recovered graph... n=$(n) - d=$(d)")
tic()
# the MDS distances:
tic()
Zmds = zeros(n,n)
Threads.@threads for i = 1:n 
        for j = 1:n
            Zmds[i,j] = norm(Xrec[:,i] - Xrec[:,j])
        end
    end
    toc()

println("Loading H")
tic()
G = dp.load_graph(data_set)
H = ld.get_dist_mat(G);
toc()

println("----------------MDS Results-----------------")
dist_max, dist, bad = dis.distortion(H, Zmds, n, 16)
println("MDS Distortion avg/max, bad = $(convert(Float64,dist)), $(convert(Float64,dist_max)), $(convert(Float64,bad))")  
mapscore = dis.map_score(H, Zmds, n, 16)
println("MAP = $(mapscore)")   
println("Bad Dists = $(bad)")
println("Dimension = $( d )") 
