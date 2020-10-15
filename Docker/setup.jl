import Pkg
Pkg.add("Pandas")
Pkg.add("JLD")
Pkg.add("PyCall")
# Force compile
using Pandas
using JLD
using Distributed

Pkg.add("ArgParse")

