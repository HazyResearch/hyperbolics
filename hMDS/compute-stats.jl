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


data_set = ARGS[1]
#fname="Xrec_dataset_",data_set,"r=",k,"prec=",prec,"tol=",tol,".jld"
X=load(fname)
println("Building recovered graph...")
tic()
Zrec = big.(zeros(n, n));
for i = 1:n
        for j = 1:n
            Zrec[i,j] = norm(Xrec[:,i] - Xrec[:,j])^2 / ((1 - norm(Xrec[:,i])^2) * (1 - norm(Xrec[:,j])^2));
        end
    end
    toc()

    
    println("----------------MDS Results-----------------")
    dist_max, dist, bad = dis.distortion(H, Zmds, n, 2)
    println("MDS Distortion avg/max, bad = $(convert(Float64,dist)), $(convert(Float64,dist_max)), $(convert(Float64,bad))")  
    mapscore = dis.map_score(H, Zmds, n, 2)
    println("MAP = $(mapscore)")   
    println("Bad Dists = $(bad)")
    println("Dimension = $( dim_mds)") 
    
    
else
    println("Dimension = 1!")
end
