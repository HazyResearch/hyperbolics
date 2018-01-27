using ArgParse
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
    end

    return parse_args(s)
end

module HP include("high_precision_polish.jl") end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    HP.serialize(parsed_args["dataset"], parsed_args["out_jld"], parsed_args["scale"],
              parsed_args["max_k"], parsed_args["prec"])
end

main()
