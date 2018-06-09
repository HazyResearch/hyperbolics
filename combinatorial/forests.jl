include("rdim.jl")

function embed_forest(dim, components, Gen_matrices, scale_cent, scale_comp)
    n_comp       = size(components)[1]
    n_comp_nodes = zeros(Int32, n_comp)

    for i = 1:n_comp
        n_comp_nodes[i] = size(components[i])[1]
    end

    n_nodes   = sum(n_comp_nodes)
    embedding = zeros(BigFloat, dim, n_nodes)

    if n_comp == 1
        centers = zeros(BigFloat, dim, 1)
    elseif n_comp == 2
        centers = zeros(BigFloat, dim, 2)
        # as far away as any two points on unit hypersphere:
        centers[1,1] = big(1.0)
        centers[1,2] = big(-1.0)
    else
        centers = place_children_codes(dim, n_comp, false, 0, Gen_matrices)
    end

    centers = scale_points(centers, scale_cent)

    # place each component:
    idx = 1
    for i = 1:n_comp
        components[i] = scale_points(components[i], scale_comp)
        for j = 1:n_comp_nodes[i]
            embedding[:, idx] = reflect_at_zero(centers[:,i], components[i][j,:])
            idx += 1
        end
    end

    return embedding'
end

function scale_points(points, scale)
    n_points = size(points)[2]

    for i = 1:n_points
        points[:,i] *= scale
    end
    return points
end

function make_gen_matrices(n_points)
    v            = Int(ceil(log2(n_points)))
    Gen_matrices = Array{Array{Float64, 2}}(v)

    for i=2:v
        n = 2^i-1
        H = zeros(i,n)
        for j=1:2^i-1
            h_col = digits(j,2,i)
            H[:,j] = h_col
        end
        Gen_matrices[i] = H
    end

    return Gen_matrices
end

