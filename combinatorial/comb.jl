# using PyCall
# using JLD
# using ArgParse
# using Pandas
import JLD
import ArgParse
import Pandas
import PyCall
@everywhere using PyCall
@everywhere using JLD
@everywhere using ArgParse
@everywhere using Pandas
@everywhere @pyimport math as pymath
@everywhere @pyimport networkx as nx
@everywhere @pyimport scipy.sparse.csgraph as csg
@everywhere @pyimport numpy as np

@everywhere unshift!(PyVector(pyimport("sys")["path"]), "")
# unshift!(PyVector(pyimport("sys")["path"]), "..")
@everywhere unshift!(PyVector(pyimport("sys")["path"]), "combinatorial")
@everywhere @pyimport utils.load_graph as lg
@everywhere @pyimport distortions as dis
@everywhere @pyimport graph_util as gu
# push!(LOAD_PATH, "combinatorial")
include("utilities.jl")
include("rdim.jl")

@everywhere py_map_row = dis.map_row 
@everywhere py_distortion_row = dis.distortion_row 

@everywhere function wrap_map_row(x,y,z,w)
    return py_map_row(x,y,z,w) 
end
@everywhere function wrap_distortion_row(x,y,z,w)
    return py_distortion_row(x,y,z,w) 
end

@everywhere function dist(u,v)
    z  = 2*norm(u-v)^2
    uu = 1 - norm(u)^2
    vv = 1 - norm(v)^2
    return acosh(1+z/(uu*vv))
end
@everywhere function dist_matrix_row(T,i)
   (n,_) = size(T)
   D = zeros(BigFloat,1,n)
   for j in 1:n
       D[1,j] = dist(T[i,:], T[j,:])
   end
   return D
end

# Parse command line arguments
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--dataset", "-d"
            help = "Dataset to embed"
            required = true
        "--eps", "-e"
            help = "Epsilon distortion"
            required = true
            arg_type = Float64
            default = 0.1
        "--dim", "-r"
            help = "Dimension r"
            arg_type = Int32
        "--get-stats", "-s"
            help = "Get statistics"
            action = :store_true
        "--embedding-save", "-m"
            help = "Save embedding to file"
        "--verbose", "-v"
            help = "Prints out row-by-row stats"
            action = :store_true
        "--scale", "-t"
            arg_type = Float64
            help = "Use a particular scaling factor"
        "--use-codes", "-c"
            help = "Use coding-theoretic child placement"
            action = :store_true
        "--stats-sample", "-z"
            help = "Number of rows to sample when computing statistics"
            arg_type = Int32
        "--precision", "-p"
            help = "Internal precision in bits"
            arg_type = Int64
            default = 256
        "--auto-tau-float", "-a"
            help = "Calculate scale assuming 64-bit final embedding"
            action = :store_true
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

println("\n\n=============================")
println("Combinatorial Embedding. Info:")
println("Data set = $(parsed_args["dataset"])")
if parsed_args["dim"] != nothing
    println("Dimensions = $(parsed_args["dim"])")
end
println("Epsilon  = $(parsed_args["eps"])")

prec = parsed_args["precision"]
setprecision(BigFloat, prec)
println("Precision = $(prec)")


if parsed_args["embedding-save"] == nothing
    println("No file specified to save embedding!")
else
    println("Save embedding to $(parsed_args["embedding-save"])")
end

# http://julia-programming-language.2336112.n4.nabble.com/copy-a-local-variable-to-all-parallel-workers-td28722.html
let GG        = lg.load_graph(parsed_args["dataset"])
@everywhere G = $GG
end
weighted = gu.is_weighted(G)
epsilon      = parsed_args["eps"]
println("\nGraph information")

# Number of vertices:
@everywhere n = G[:order]();
println("Number of vertices = $(n)");

# Number of edges
num_edges = G[:number_of_edges]();
println("Number of edges = $(num_edges)");

# We'll use the BFS tree here - if G is already a tree
#  this won't change anything, but if it's not we'll get a tree
#  Also lets us use G.successors, parent, etc. function
@everywhere root, d_max   = gu.max_degree(G)
@everywhere G_BFS   = gu.get_BFS_tree(G, root)

# A few statistics
@everywhere n_bfs   = G_BFS[:order]();
degrees = G_BFS[:degree]();
println("Max degree = $(d_max)")

# Adjacency matrices for original graph and for BFS
@everywhere adj_mat_original    = nx.to_scipy_sparse_matrix(G, 0:n-1)
adj_mat_bfs         = nx.to_scipy_sparse_matrix(G_BFS, 0:n_bfs-1)

# Perform the embedding
println("\nPerforming the embedding")
tic()

if parsed_args["scale"] != nothing
    tau = big(parsed_args["scale"])
elseif parsed_args["auto-tau-float"]
    path_length  = nx.dag_longest_path_length(G_BFS)
    r = big(1-eps(BigFloat)/2)
    m = big(log((1+r)/(1-r)))
    tau = big(m/(1.3*path_length))
else
    tau = get_emb_par(G_BFS, 1, epsilon, weighted)
end

let _tau        = tau
@everywhere tau = $_tau
end

# Print out the scaling factor we got
println("Scaling factor tau = $(convert(Float64,tau))")

use_codes = false
if parsed_args["use-codes"]
    println("Using coding theoretic child placement")
    use_codes = true
else
    println("Using uniform sphere child placement")
end

if parsed_args["dim"] != nothing && parsed_args["dim"] != 2
    dim = parsed_args["dim"]
    T = hyp_embedding_dim(G_BFS, root, epsilon, weighted, dim, tau, d_max, use_codes)
else
    T = hyp_embedding(G_BFS, root, epsilon, weighted, tau)
end
toc()

let _T = T
@everywhere T = $_T
end

# Save the embedding:
if parsed_args["embedding-save"] != nothing
    JLD.save(string(parsed_args["embedding-save"],".jld"), "T", T);
    df = DataFrame(convert(Array{Float64,2},T))
    # save tau also:
    df["tau"] = convert(Float64, tau)
    to_csv(df, parsed_args["embedding-save"])
end

if parsed_args["get-stats"]
    tic()
    println("\nComputing quality statistics")
    # The rest is statistics: MAP, distortion
    maps = 0;
    wc = 1;
    d_avg = 0;

    # In case we want to sample the rows of the matrix:
    if parsed_args["stats-sample"] != nothing
        samples = min(parsed_args["stats-sample"], n_bfs)
        println("Using $samples sample rows for statistics")
    else
        @everywhere samples = n_bfs
    end
    @everywhere sample_nodes = randperm(n_bfs)[1:samples]

    _maps   = zeros(samples)
    _d_avgs = zeros(samples)
    _wcs    = zeros(samples)

    # Threads.@threads for i=1:samples

    # compute stats for row i
    @everywhere function compute_row_stats(i)
        # the real distances in the graph
        # true_dist_row = vec(csg.dijkstra(adj_mat_original, indices=[sample_nodes[i]-1], unweighted=(!weighted), directed=false))
        true_dist_row = vec(csg.dijkstra(adj_mat_original, indices=[sample_nodes[i]-1], unweighted=true, directed=false))

        # the hyperbolic distances for the points we've embedded
        hyp_dist_row = convert(Array{Float64},vec(dist_matrix_row(T, sample_nodes[i])/tau))

        # this is this row MAP
        # TODO: should this be n_bfs instead of n? n might include points that weren't embedded?
        # curr_map  = dis.map_row(true_dist_row, hyp_dist_row[1:n], n_bfs, sample_nodes[i]-1)
        curr_map  = wrap_map_row(true_dist_row, hyp_dist_row[1:n], n_bfs, sample_nodes[i]-1)
        # _maps[i]  = curr_map

        # print out current and running average MAP
        # if parsed_args["verbose"]
        #     println("Row $(sample_nodes[i]), current MAP = $(curr_map)")
        # end

        # these are distortions: worst cases (contraction, expansion) and average
        # mc, me, avg, bad = dis.distortion_row(true_dist_row, hyp_dist_row[1:n] ,n,sample_nodes[i]-1)
        mc, me, avg, bad = wrap_distortion_row(true_dist_row, hyp_dist_row[1:n] ,n,sample_nodes[i]-1)
        # _wcs[i]  = mc*me

        # _d_avgs[i] = avg
        return (curr_map, avg, mc*me)
    end
    # reduction function
    f = (a,b) -> (a[1]+b[1], a[2]+b[2], max(a[3],b[3]))
    # (maps, d_avg, wc) = @parallel f for i=1:samples
    #     compute_row_stats(i)
    # end
    # Clean up
    (maps, d_avg, wc) = reduce(f, pmap(compute_row_stats, 1:samples))
    # maps  = sum(_maps)
    # d_avg = sum(_d_avgs)
    # wc    = maximum(_wcs)

    if weighted
        println("Note: MAP is not well defined for weighted graphs")
    end

    # Final stats:
    println("Final MAP = $(maps/samples)")
    println("Final d_avg = $(d_avg/samples), d_wc = $(wc)")
    println("Sum MAP = $(maps)")
    println("Sum d_avg = $(d_avg)")
    toc()
end
