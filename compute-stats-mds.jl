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
function distortion_entry(h,h_rec,me,mc)
    avg = abs(h_rec - h)/h                            
    if h_rec/h > me me = h_rec/h end
    if h/h_rec > mc mc = h/h_rec end
    return (avg,me,mc)
end
## function map_row(H1, H2, n, row; verbose=true)
##     edge_mask = (H1 == 1.0)
##     m         = sum(edge_mask)
##     if verbose print("\t There are $(m) edges for $(row) of $(n)") end
##     sorted_dist = sortperm(H2)
##     ## if verbose
##     ##     print(f"\t $(sorted_dist[1:5]) vs. $(np.array(range(n))[edge_mask]}")
##     ##     print(f"\t {d[sorted_dist[1:5]]} vs. {H1[edge_mask]}")
##     ## end
##     precs       = zeros(m)    
##     n_correct   = 0
##     j = 0
##     # skip yourself, you're always the nearest guy    
##     for i=2:n 
##         if edge_mask[sorted_dist[i]]
##             n_correct += 1
##             precs[j] = n_correct/float(i)
##             j += 1
##             if j == m break end
##         end
##     end
##     return sum(precs)/m 
## end


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
        if j == i continue end
        if isnan(hrow_i[j]) || isnan(H[i,j]) || H[i,j] == 0
            bad[i] += 1
            continue
        end
        #if j == i || !entry_is_good(H[i,j], hrow_i[j]) continue end
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
println("----------------MDS Results-----------------")
#dist_max, dist, bad = dis.distortion(H, Zmds, n, 16)
#println("MDS Distortion avg/max, bad = $(dist), $(dist_max), $(bad)")
println("MDS Distortion avg/max, bad = $(sum(avgs)), $(maximum(mes)*maximum(mcs)) $(sum(bad))")  
#mapscore  = dis.map_score(H, Zmds, n, 16)
#println(maps)
mapscore2 = sum(maps)/n 
println("MAP = $(mapscore2)")   
#println("Bad Dists = $(bad)")
println("Dimension = $( d )") 
