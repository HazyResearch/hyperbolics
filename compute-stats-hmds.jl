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
fname    = ARGS[2]
scale    = parse(Float64, (ARGS[3]))
println("Loading")
tic()
X_f=load(fname)
Xrec = X_f["Xrec"]
toc()
(d,n) = size(Xrec)
println("Building recovered graph... n=$(n) - d=$(d)")
# the MDS distances:
tic()
 println("Building recovered graph...")
 tic()
 Hrec = zeros(n, n)

Threads.@threads for i = 1:n
        for j = 1:n
            v = norm(Xrec[:,i] - Xrec[:,j])^2 / ((1 - norm(Xrec[:,i])^2) * (1 - norm(Xrec[:,j])^2))
	    Hrec[i,j] = acosh(1 +2*max(v,0.0))/scale
        end 
end
toc()
   

println("Loading H")
tic()
G = dp.load_graph(data_set)
H = ld.get_dist_mat(G);
toc()

println("----------------hMDS Results-----------------")
dist_max, dist, good = dis.distortion(H, Hrec, n, 16)
println("Distortion avg/max, bad = $(convert(Float64,dist)), $(convert(Float64,dist_max)), $(convert(Float64,good))")  
mapscore = dis.map_score(H, Hrec, n, 16) 
println("MAP = $(mapscore)")   
println("Dimension = $(found_dimension)")
toc() 
