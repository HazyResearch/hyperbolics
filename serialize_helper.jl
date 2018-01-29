using ArgParse
using JSON
using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport distortions as dis

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--prec"
           help = "precision"
           arg_type = Int
           default = 1024
        "--max_k" 
            help = "maximum k"
            arg_type = Int
            default = 200
        "--scale"
            arg_type = Float64
            default  = 1.0
        "dataset"
            arg_type = Int
            help = "Dataset"
            required = true
        "out_jld"
            required = true
            help = "outfile"
        "--stats-file"
           help="Output stat file"
           default ="stats.out"
        "--low-precision"
           help="Do low precision solves too"
           default = false
        "--load-file"
           help="load a file"
    end

    return parse_args(s)
end

module HP include("high_precision_polish.jl") end


function compute_metrics(H, Xrec, name, parsed_args)
    scale = parsed_args["scale"]
    println("Building recovered graph for $(name)...")
    (n,n) = size(H)
    tic()
    Zrec = big.(zeros(n, n));
    for i = 1:n
        for j = 1:n
            # Note the graph is _not_ renormalized to hyperbolic space. TODO. Fix this!
            #Zrec[i,j] = norm(Xrec[:,i] - Xrec[:,j])^2 / ((1 - norm(Xrec[:,i])^2) * (1 - norm(Xrec[:,j])^2));
            Zrec[i,j] = norm(Xrec[:,i] - Xrec[:,j])^2 
        end
    end
    toc()

    println("Getting metrics")
    tic()
    Hrec = Float64.(acosh.(1+2*Zrec)/scale)
    
    dist_max, dist, good = dis.distortion(H, Hrec, n, 2)
    println("\tDistortion avg/max, bad = $(convert(Float64,dist)), $(convert(Float64,dist_max)), $(convert(Float64,good))")
    mapscore = dis.map_score(Int64.(H), Hrec, n, 2) 
    println("\tMAP = $(mapscore)")
    d   = Dict("MAP" => Float64(mapscore), "avg_d" => Float64(dist), "max_d" => Float64(dist_max), "good" => Float64(good))
    ret = Dict("experiment" => name, "values" => d, "rec_size" => size(Xrec) )
    println(ret)
    return ret
end

function get_points(_vals, _eigs)    
    idx       = sortperm(_vals, rev=true)
    eigs      = _eigs[:,idx]
    vals      = _vals[idx]
    pos_idx   = vals .> 0

    eigs      = eigs[:,pos_idx]
    vals      = vals[pos_idx]
    x_recs    = eigs*diagm(sqrt.(vals))
    return x_recs
end

function print_stats(H,f,_vals, _eigs,name, parsed_args)
    x_recs    = get_points(_vals,_eigs)
    for r in [2,5,10,20,50,100,200]
        _r  = min(r,size(x_recs)[2])
        t_r = _r
        stats_line = compute_metrics(H, x_recs[:,1:_r]', "$(name)_$(r)_$(_r)", parsed_args)
        write(f,"$(json(stats_line))\n")
        if _r < r break end
    end
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    
    if parsed_args["load-file"] == nothing
        (H,_vals,_eigs,T) = HP.serialize(parsed_args["dataset"], parsed_args["out_jld"], parsed_args["scale"],
                                         parsed_args["max_k"], parsed_args["prec"])
    else
        load_file = parsed_args["load-file"]
        println("Loading File! $(load_file)")
        setprecision(BigFloat, parsed_args["prec"])
        f = JLD.load(load_file)
        (_vals,_eigs,T) = (f["M_val"], f["M_eigs"], f["M_T"])
        H = HP.load_graph(parsed_args["dataset"])
        println("Loaded")
    end
    stat_file = parsed_args["stats-file"]
    prec      = parsed_args["prec"]
    open(stat_file, "w") do f
        print_stats(H,f,_vals, _eigs, "Precision_$(prec)", parsed_args)
        # Low precision version
        if parsed_args["low-precision"]
            ST              = SymTridiagonal(Float64.(diag(T)), Float64.(diag(T,1)))
            (t_vals,t_eigs) = eigs(ST)
            print_stats(H,f,_vals,_eigs,"Low_Precision", parsed_args)
        end
    end
end

main()
