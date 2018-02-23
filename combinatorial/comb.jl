using PyCall
using JLD
using ArgParse
@pyimport networkx as nx
@pyimport scipy.sparse.csgraph as csg
@pyimport numpy as np

unshift!(PyVector(pyimport("sys")["path"]), "..")
@pyimport load_graph as lg
@pyimport distortions as dis
include("utilities.jl")

# Slight overkill, but should work for eps=0.1
setprecision(BigFloat, 16384)

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
        "--get-stats", "-s"
            help = "Get statistics"
            action = :store_true            
        "--embedding-save", "-m"
            help = "Save embedding to file"
        "--verbose", "-v"
            help = "Prints out row-by-row stats"
            action = :store_true                    
    end
    return parse_args(s)
end

parsed_args = parse_commandline()

println("Combinatorial Embedding. Info:")
println("Data set = $(parsed_args["dataset"])")
println("Epsilon  = $(parsed_args["eps"])")

if parsed_args["embedding-save"] == nothing
    println("No file specified to save embedding!")
else
    println("Save embedding to $(parsed_args["embedding-save"])")
end

G        = lg.load_graph(parsed_args["dataset"])
weighted = false;
eps      = parsed_args["eps"] 

println("\nGraph information")
# Number of vertices:
n = G[:order]();
println("Number of vertices = $(n)");

# Number of edges
num_edges = G[:number_of_edges]();
println("Number of edges = $(num_edges)");
edges_weights = []

# We'll use the BFS tree here - if G is already a tree
#  this won't change anything, but if it's not we'll get a tree
#  Also lets us use G.successors, parent, etc. function
G_BFS   = nx.bfs_tree(G,0)

# A few statistics
n_bfs   = G_BFS[:order]();
degrees = G_BFS[:degree]();
cd      = collect(degrees);
d_max   = maximum([cd[i][2] for i in 1:n])
println("Max degree = $(d_max)")

# Adjacency matrices for original graph and for BFS
adj_mat_original    = nx.to_scipy_sparse_matrix(G, 0:n-1)
adj_mat_bfs         = nx.to_scipy_sparse_matrix(G_BFS, 0:n_bfs-1)

# Perform the embedding
println("\nPerforming the embedding")
tic()
if weighted
    (T, tau) = hyp_embedding(G_BFS, eps, weighted, edges_weights, G_S)
else
    (T, tau) = hyp_embedding(G_BFS, eps, weighted, edges_weights, G_BFS)
end
toc()

# Print out the scaling factor we got
println("Scaling factor tau = $(convert(Float64,tau))")

# Save the embedding:
if parsed_args["embedding-save"] != nothing
    save(string(parsed_args["embedding-save"],".jld"), "T", T);
end

if parsed_args["get-stats"]
    println("\nComputing quality statistics")
    # The rest is statistics: MAP, distortion
    maps = 0;
    wc = 1;
    d_avg = 0;

    # In case we want to sample the rows of the matrix:
    samples = min(1000,n)
    sample_nodes = randperm(n)[1:samples]

    for i=1:samples
        # the real distances in the graph
        true_dist_row = vec(csg.dijkstra(adj_mat_original, indices=[sample_nodes[i]-1], unweighted=true, directed=false))

        # the hyperbolic distances for the points we've embedded
        hyp_dist_row = convert(Array{Float64},vec(dist_matrix_row(T, sample_nodes[i])/tau))

        # this is this row MAP
        curr_map  = dis.map_row(true_dist_row, hyp_dist_row[1:n], n, sample_nodes[i]-1)
        maps += curr_map

        # print out current and running average MAP
        if parsed_args["verbose"]
            println("Row $(sample_nodes[i]), current MAP = $(curr_map)")        
            println("Row $(sample_nodes[i]), running MAP = $(maps/i)") 
        end

        # these are distortions: worst cases (contraction, expansion) and average
        mc, me, avg, bad = dis.distortion_row(true_dist_row, hyp_dist_row[1:n] ,n,sample_nodes[i]-1)
        if mc*me > wc
            wc = mc*me
        end
        d_avg += avg;
        
        if parsed_args["verbose"]
            println("Row $(sample_nodes[i]), current d_avg = $(d_avg/i), current d_wc = $(wc)")
        end
    end

    # Final stats:
    println("Final MAP = $(maps/samples)")
    println("Final d_avg = $(d_avg/n), d_wc = $(wc)")
end


