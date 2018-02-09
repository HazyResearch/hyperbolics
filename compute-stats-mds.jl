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

data_set   = parse(Int32,ARGS[1])
fname      = ARGS[2]
to_sample  = parse(Int32, ARGS[3])

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
function map_row(H1, H2, n, row; verbose=false)
    edge_mask = (H1 .== 1.0)
    m         = sum(edge_mask)
    if verbose println("\t There are $(m) edges for $(row) of $(n)") end
    sorted_dist = sortperm(H2)
    if verbose
        println("\t $(sorted_dist[1:5]) vs. $(collect(1:n)[edge_mask])")
        println("\t $(H2[sorted_dist[1:5]]) vs. $(H1[edge_mask])")
    end
    precs       = zeros(m)    
    n_correct   = 0
    j           = 1
    # skip yourself, you're always the nearest one    
    for i=2:n 
        if edge_mask[sorted_dist[i]]
            n_correct += 1
            precs[j]   = n_correct/float(i-1)
            j         += 1
            if j > m break end
        end
    end
    if verbose println("\t precs=$(precs)") end
    return sum(precs)/float(m) 
end


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
println("Using $(Threads.nthreads()) threads")
hrow_i = zeros(Threads.nthreads(),n)
t1   = time_ns()
indexes = shuffle(collect(1:n))

Threads.@threads for i = indexes[1:to_sample]
    t = Threads.threadid()
    for j = 1:n
        hrow_i[t,j] = norm(Xrec[:,i] - Xrec[:,j])
        if j == i continue end
        if isnan(hrow_i[t,j]) || isnan(H[i,j]) || H[i,j] == 0
            bad[i] += 1
            continue
        end
        #if j == i || !entry_is_good(H[i,j], hrow_i[j]) continue end
        (avg,me,mc) = distortion_entry(H[i,j], hrow_i[t,j], mes[i], mcs[i])
        avgs[i] += avg/(float(to_sample*(n-1)))
        mes[i]   = me
        mcs[i]   = mc
    end
    maps[i] = map_row(H[i,:], hrow_i[t,:], n, i)
    # Python call. watch the indexing.
    #print(".")
    if i % 10 == 0 ccall(:jl_,Void,(Any,), "$(i) done $((time_ns() - t1)/1e9)") end
end
toc()


println("Loading H")
println("----------------MDS Results-----------------")
#dist_max, dist, bad = dis.distortion(H, Zmds, n, 16)
#println("MDS Distortion avg/max, bad = $(dist), $(dist_max), $(bad)")
println("MDS Distortion avg/max, bad = $(sum(avgs)), $(maximum(mes)*maximum(mcs)) $(sum(bad))")  
#mapscore  = dis.map_score(H, Zmds, n, 16)
#println(maps)
mapscore2 = sum(maps)/to_sample 
println("MAP = $(mapscore2) sampled=$(to_sample)")   
#println("Bad Dists = $(bad)")
println("Dimension = $( d )") 
