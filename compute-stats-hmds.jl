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
# the MDS distances:
function distortion_entry(h,h_rec,me,mc)
    avg = abs(h_rec - h)/h                            
    if h_rec/h > me me = h_rec/h end
    if h/h_rec > mc mc = h/h_rec end
    return (avg,me,mc)
end

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
#Hrec = zeros(n, n)
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
        hrow_i[j]   = norm(Xrec[:,i] - Xrec[:,j])^2/((1 -norm(Xrec[:,i])^2)*(1- norm(Xrec[:,j])^2))
        #if j == i || !entry_is_good(H[i,j], hrow_i[j]) continue end
        if j == i continue end
        (avg,me,mc) = distortion_entry(H[i,j], acosh(1+2*max(hrow_i[j],0))/scale, mes[i], mcs[i])
        avgs[i] += avg/(float(n*(n-1)))
        mes[i]   = me
        mcs[i]   = mc
        #Hrec[i,j] = acosh(1+2*hrow_i[j]) # TO REMOVE
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
#dist_max, dist, good = dis.distortion(H, Hrec/scale, n, 16)
#println("Distortion avg/max, bad = $(convert(Float64,dist)), $(convert(Float64,dist_max)), $(convert(Float64,good))")
println("HMDS Distortion avg/max, bad = $(sum(avgs)), $(maximum(mes)*maximum(mcs)) $(sum(bad))")  


#mapscore = dis.map_score(H, Hrec, n, 16) 
#println("MAP = $(mapscore)")

mapscore2 = sum(maps)/n 
println("MAP = $(mapscore2)")   

println("Dimension = $(d)")
toc() 
