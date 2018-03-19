# distances.jl
# functions for computing distances; particularly needed in final statistics computation
# separatedly into separate file for ease of including on multiple processes


# module Distances
# export dist, dist_matrix_row

# Hyperbolic distance d_H(u,v)
function dist(u,v)
    z  = 2*norm(u-v)^2
    uu = 1 - norm(u)^2
    vv = 1 - norm(v)^2
    return acosh(1+z/(uu*vv))
end

# Compute distances from i to all others
function dist_matrix_row(T,i)
    (n,_) = size(T)
    D = zeros(BigFloat,1,n)
    for j in 1:n
        D[1,j] = dist(T[i,:], T[j,:])
    end
    return D
end

# end
