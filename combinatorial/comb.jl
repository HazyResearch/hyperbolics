using PyCall
using JLD
using ArgParse
using Pandas
@pyimport networkx as nx
@pyimport scipy.sparse.csgraph as csg
@pyimport numpy as np

unshift!(PyVector(pyimport("sys")["path"]), "")
# unshift!(PyVector(pyimport("sys")["path"]), "..")
unshift!(PyVector(pyimport("sys")["path"]), "combinatorial")
@pyimport utils.load_graph as lg
@pyimport utils.distortions as dis
@pyimport graph_util as gu
include("utilities.jl")
include("rdim.jl")
include("forests.jl")
include("distances.jl")


# Parse command line arguments
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--dataset", "-d"
            help = "Dataset to embed"
            required = true
        "--eps", "-e"
            help = "Epsilon distortion"
            # required = true
            arg_type = Float64
            default = 0.1
        "--dim", "-r"
            help = "Dimension r"
            arg_type = Int64
            default = 2
        "--precision", "-p"
            help = "Internal precision in bits"
            arg_type = Int64
            default = 256
        "--scale", "-t"
            arg_type = Float64
            help = "Use a particular scaling factor"
        "--auto-tau-float", "-a"
            help = "Calculate scale assuming 64-bit final embedding"
            action = :store_true
        "--save-embedding", "-m"
            help = "Save embedding to file"
       # "--save-embedding-forest", "-mf"
	   # help = "Save forest embedding to file"
        "--verbose", "-v"
            help = "Prints out row-by-row stats"
            action = :store_true
        "--use-codes", "-c"
            help = "Use coding-theoretic child placement"
            action = :store_true
        "--stats-sample", "-z"
            help = "Number of rows to sample when computing statistics"
            arg_type = Int32
        "--save-distances", "-y"
            help = "Save the distance matrix"
        "--get-stats", "-s"
            help = "Get statistics"
            action = :store_true
        "--procs", "-q"
            help = "Number of processes to use"
            arg_type = Int32
        "--forest", "-f"
            help = "Forest embedding; use all the edgelists in the given folder"
            action = :store_true
        "--visualize", "-w"
            help = "Visualize the embedding (only for 2 dimensions)"
            action = :store_true
    end
    return parse_args(s)
end

function do_embedding(parsed_args)
    println("\n=============================")
    println("Combinatorial Embedding. Info:")
    println("Data set = $(parsed_args["dataset"])")
    println("Dimensions = $(parsed_args["dim"])")

    # visualization checks:
    visualize = false
    if parsed_args["visualize"]
        if parsed_args["dim"] > 2
            println("Error: visualization not supported for more than 2 dimensions!")
            exit()
        end
        visualize = true
    end

    prec = parsed_args["precision"]
    setprecision(BigFloat, prec)
    println("Precision = $(prec)")


    if parsed_args["save-embedding"] == nothing
        println("No file specified to save embedding!")
    else
        println("Save embedding to $(parsed_args["save-embedding"])")
    end

    G        = lg.load_graph(parsed_args["dataset"])
    weighted = gu.is_weighted(G)
    println("\nGraph information")

    # Number of vertices:
    n = G[:order]();
    println("Number of vertices = $(n)");

    # Number of edges
    num_edges = G[:number_of_edges]();
    println("Number of edges = $(num_edges)");

    # We'll use the BFS tree here - if G is already a tree
    #  this won't change anything, but if it's not we'll get a tree
    #  Also lets us use G.successors, parent, etc. function
    root, d_max   = gu.max_degree(G)
    root = 0
    G_BFS   = gu.get_BFS_tree(G, root)

    # A few statistics
    n_bfs   = G_BFS[:order]();
    degrees = G_BFS[:degree]();
    path_length  = nx.dag_longest_path_length(G_BFS)
    println("Max degree = $(d_max), Max path = $(path_length)")

    # Adjacency matrices for original graph and for BFS
    adj_mat_original    = nx.to_scipy_sparse_matrix(G, 0:n-1)
    adj_mat_bfs         = nx.to_scipy_sparse_matrix(G_BFS, 0:n_bfs-1)

    # Perform the embedding
    println("\nPerforming the embedding")
    tic()

    if parsed_args["scale"] != nothing
        tau = big(parsed_args["scale"])
    elseif parsed_args["auto-tau-float"]
        r = big(1-eps(BigFloat)/2)
        m = big(log((1+r)/(1-r)))
        tau = big(m/(1.3*path_length))
    elseif parsed_args["eps"] != nothing
        println("Epsilon  = $(parsed_args["eps"])")
        epsilon      = parsed_args["eps"]
        tau = get_emb_par(G_BFS, 1, epsilon, weighted)
    else
        tau = 1.0
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
        T = hyp_embedding_dim(G_BFS, root, weighted, dim, tau, d_max, use_codes)
    else
        T = hyp_embedding(G_BFS, root, weighted, tau, visualize)
    end
    toc()

    # Save the embedding:
    if parsed_args["save-embedding"] != nothing
        # JLD.save(string(parsed_args["save-embedding"],".jld"), "T", T);
        df = DataFrame(convert(Array{Float64,2},T))
        # save tau also:
        df["tau"] = convert(Float64, tau)
        to_csv(df, parsed_args["save-embedding"])
    end

    if parsed_args["get-stats"]
    include(pwd() * "/combinatorial/distances.jl")
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
            samples = n_bfs
        end
        sample_nodes = randperm(n_bfs)[1:samples]

        _maps   = zeros(samples)
        _d_avgs = zeros(samples)
        _wcs    = zeros(samples)

        for i in 1:n
            # the real distances in the graph
            true_dist_row = vec(csg.dijkstra(adj_mat_original, indices=[sample_nodes[i]-1], unweighted=(!weighted), directed=false))

            # the hyperbolic distances for the points we've embedded
            hyp_dist_row = convert(Array{Float64},vec(dist_matrix_row(T, sample_nodes[i])/tau))

            # this is this row MAP
            # TODO: n=n_bfs for the way we're currently loading data, but be careful in future
            curr_map  = dis.map_row(true_dist_row, hyp_dist_row[1:n], n, sample_nodes[i]-1)
            _maps[i]  = curr_map

            # print out current and running average MAP
            if parsed_args["verbose"]
                println("Row $(sample_nodes[i]), current MAP = $(curr_map)")
            end

            # these are distortions: worst cases (contraction, expansion) and average
            mc, me, avg, bad = dis.distortion_row(true_dist_row, hyp_dist_row[1:n] ,n,sample_nodes[i]-1)
            _wcs[i]  = mc*me

            _d_avgs[i] = avg
        end
        # Clean up
        maps  = sum(_maps)
        d_avg = sum(_d_avgs)
        wc    = maximum(_wcs)

        if weighted
            println("Note: MAP is not well defined for weighted graphs")
        end

        # Final stats:
        println("Final MAP = $(maps/samples)")
        println("Final d_avg = $(d_avg/samples), d_wc = $(wc)")
    end

    if parsed_args["save-distances"] != nothing
        println("\nSaving embedded distance matrix")

        srand(0) # shuffle data so we can take terminate early and take sampled statistics

        if parsed_args["procs"] != nothing
            addprocs(parsed_args["procs"]-1)
        end

        # *****Multiprocessing memory sharing*****
        # @everywhere using Distances: dist, dist_matrix_row
        tic()
        @everywhere include(pwd() * "/combinatorial/distances.jl")

        @eval @everywhere T = $T
        @eval @everywhere tau = $tau

        # set global precision
        @eval @everywhere prec = $prec
        @everywhere setprecision(BigFloat, prec)

        @sync @everywhere _dist_matrix_row = i -> convert(Array{Float64},vec(dist_matrix_row(T, i)/tau))

        println("Finished multiprocess memory sharing")
        toc()
        # *****End memory sharing*****

        function _compute_and_save_rows(i, rows)
            tic()
            # *****Multiprocessing solution*****
            hyp_dist_mat = hcat(pmap(_dist_matrix_row, rows)...)'
            # *****End multiprocessing solution*****

            # TODO: enable either multiprocessing vs multithreading with a flag to pass in; set JULIA_NUM_THREADS with ENV[] call

            # *****Multithreading solution*****
            # include(pwd() * "/combinatorial/distances.jl")
            # hyp_dist_mat = zeros(n_bfs, n_bfs)
            # Threads.@threads for i=1:n_bfs
            # # for i in 1:n_bfs
            #    # the hyperbolic distances for the points we've embedded
            #     hyppython3 scripts/comb_stats.py weighted_wordnet wordnet_closure -t 1.0 -r 2 -p 32768^Cq 96_dist_row = convert(Array{Float64},vec(dist_matrix_row(T, i)/tau))

            #     hyp_dist_mat[i,:] = hyp_dist_row
            # end
            # *****End multithreading solution*****

            println("Finished computing rows of distance matrix, chunk $(i)")
            toc()

            tic()
            D = DataFrame(hyp_dist_mat, index=rows-1)
            to_csv(D, parsed_args["save-distances"] * ".$(i)")
            println("Finished writing rows of distance matrix")
            toc()
        end

        # Random chunk and compute
        sample_nodes = randperm(n_bfs)
        chunk_sz = 1000
        chunk_i = 0
        while chunk_i * chunk_sz < n_bfs
            chunk_start = 1 + chunk_i * chunk_sz
            chunk_end = min(chunk_start+chunk_sz-1, n_bfs)
            chunk_rows = sample_nodes[chunk_start:chunk_end]
            _compute_and_save_rows(chunk_i, chunk_rows)
            chunk_i += 1
        end

    end
    println()

    return T, tau
end

parsed_args = parse_commandline()

# if we are embedding a forest, get each edge list in the current directory
if parsed_args["forest"]
    edges_folder = parsed_args["dataset"]
    files        = readdir(edges_folder)
    n_files      = size(files)[1]
    components   = Array{Array{}}(n_files);
    tau_components = Array{Float64}(n_files);

    for i = 1:n_files
        parsed_args["dataset"] = string(edges_folder, '/', files[i])
        components[i], tau_components[i] = do_embedding(parsed_args)
    end
    println(tau_components)
    println(files)
    
    Gen_matrices = make_gen_matrices(n_files)
    F = embed_forest(parsed_args["dim"], components, Gen_matrices, big(0.9), big(1.0))
    println(tau_components)
    # Save the forest embedding:
    if parsed_args["save-embedding"] != nothing
        dforest = DataFrame(convert(Array{Float64,2},F))
	dcomp_tau_components = DataFrame(convert(Array{Float64,1},tau_components))
        # save tau for each cc and corresponding file name
	dfiles = DataFrame(convert(Array{String},files))
        to_csv(dforest, parsed_args["save-embedding"])
        to_csv(dcomp_tau_components, "tau_test.txt")
	to_csv(dfiles, "cc_test.txt")
    end
    # debug:
    # n_points = size(F)[1]
    # for i=1:n_points
        # println(F[i,:])
    # end
else
    do_embedding(parsed_args)
end
