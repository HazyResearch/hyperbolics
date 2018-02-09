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
Xrec = X_f["XRec"]
toc()
(d,n) = size(Xrec)
println("Building recovered graph... n=$(n) - d=$(d)")
# the MDS distances:
tic()
 println("Building recovered graph...")
 tic()
 Hrec = zeros(n, n)
tic()
G = dp.load_graph(data_set)
H = ld.get_dist_mat(G);
toc()

tic()
avgs = zeros(n)
mes  = zeros(n)
mcs  = zeros(n)
maps = zeros(n)
bad  = zeros(n)
Threads.@threads for i = 1:n
    hrow_i = zeros(n) # thread local?
    tic()
    for j = 1:n
        hrow_i[j]   = norm(Xrec[:,i] - Xrec[:,j])
        #if j == i || !entry_is_good(H[i,j], hrow_i[j]) continue end
        if j == i continue end
        (avg,me,mc) = distortion_entry(H[i,j], hrow_i[j], mes[i], mcs[i])
        avgs[i] += avg/(float(n*(n-1)))
        mes[i]   = me
        mcs[i]   = mc
    end
    #maps[i] = map_row(H[i,:], hrow_i, n, i)
    # Python call. watch the indexing.
    maps[i] = dis.map_row(H[i,:], hrow_i, n, i-1)
    print(".")
    if i % 10 == 0 toc() end
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
println("Dimension = $(d)")
toc() 
