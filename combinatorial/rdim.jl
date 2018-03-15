    # rdim.jl
# various utilities for higher-dimensional combinatorial embeddings

# rotate the set of points so that the first vector 
#  coincides with the starting point sp
#  N = dimension, K = # of points 
function rotate_points(points, sp, N, K)
    pts  = zeros(BigFloat, N, K)
    x    = points[:,1]
    y    = sp
    
    # we'll rotate x to y
    u    = big.(x/norm(x))
    v    = big.(y-u'*y*u)
    v    = big.(v/norm(v))
    cost = big.(x'*y/norm(x)/norm(y))

    # no rotation needed:
    if (big(1)-cost^2) <= big(0)
        return points    
    end
    sint = sqrt(big(1)-cost^2)
    
    R = big.(eye(length(x)))-u*u'-v*v' + [u v]*[cost -sint;sint cost]*[u v]'
    
    for i=1:K
        pts[:,i] = R*points[:,i];    
    end
         
    return pts
end

# spherical coodinates: get angles from a set of Euclidean coord.
function angle_from_coord(x, N)
    r = big(norm(x))
    ang = zeros(BigFloat, N-1, 1)
    
    for i=1:N-2
        ang[i] = big(acos(x[i]/norm(x[i:N])));   
    end
    
    if x[N] >= 0
        ang[N-1] = big(acos(x[N-1]/norm(x[N-1:N])))
    else
        ang[N-1] = big(2)*big(pi) - big(acos(x[N-1]/norm(x[N-1:N])))
    end
    
    return ang
end

# spherical coodinates: get Euclidean coord. from a set of points
function coord_from_angle(ang, N)
    point = zeros(BigFloat, N, 1)

    for i=1:N-1
        if i==1 
            point[i] = big(cos(ang[i]))
        else()
            point[i] = big(prod(sin.(ang[1:i-1])))
            point[i] = big(point[i] * cos(ang[i]))
        end
    end
    
    point[N] = big(prod(sin.(ang)))
    return point
end


# algorithm to place a set of points on the n-dimensional unit sphere based on coding theory
#  The idea is that we will place points to be vertices of a hypercube inscribed
#  into the unit sphere, ie, with coordinates (a_1/sqrt(n),...,a_n/sqrt(n))
#  where n is the dimension and a_1,...,a_n \in \{-1,1}.
#  
#  It's easy to show that if d = Hamming_distance(a,b), then the Euclidean distance
#  between the two vectors is 2sqrt(d/n). We maximize d by using a code.
#  In this case, our code is the simplex code, with length 2^z-1, and dimension z
#  In this code, the Hamming distance between any pair of vectors is 2^{z-1}.
#  Dimension z means that we have 2^z codewords, so we can place up to 2^z children.
#  one additional challenge is that our dimensions might be too large, e.g., 
#  dim > 2^z-1 for some number of children. Then we generate a codeword and repeat it
#  Note also that the generator matrix for the simplex code is the parity check matrix
#  of the Hamming code, which we precompute for all the z's of interest 
function place_children_codes(dim, c, use_sp, sp, Gen_matrices)
    r = Int(ceil(log2(c)))
    n = 2^r-1
    
    G = Gen_matrices[r]
    
    # generate the codewords using our matrix
    C = zeros(BigFloat, c, dim)
    for i=0:c-1
        # codeword generated from matrix G:
        cw = (digits(i,2,r)'*G).%(2)

        rep = Int(floor(dim/n))
        for j=1:rep
            # repeat it as many times as we can
            C[i+1,(j-1)*n+1:j*n] = big.(cw');  
        end
        rm = dim-rep*n
        if rm > 0
            C[i+1,rep*n+1:dim] = big.(cw[1:rm]')
        end
    end    
    
    # inscribe the unit hypercube vertices into unit hypersphere
    points = (big(1)/sqrt(dim)*(-1).^C)'
        
    # rotate to match the parent, if we need to
    if use_sp == true
        points = rotate_points(points, sp, dim, c)
    end
    
    return points
end


# algorithm to place a set of points uniformly on the n-dimensional unit sphere
#  see: http://www02.smt.ufrj.br/~eduardo/papers/ri15.pdf
#  the idea is to try to tile the surface of the sphere with hypercubes
#  works well for larger numbers of points
# what we'll actually do is to build it once with a large number of points
#  and then sample for this set of points
function place_children(dim, c, use_sp, sp, sample_from, sb)
    # this notation comes from the paper
    N = dim
    K = c
    
    if sample_from
        _, K_large = size(sb)
        points = zeros(BigFloat, N, K)

        for i=0:K-2
            points[:,1+i] = sb[:, Int(floor(K_large/(K-1)))*i+1];  
        end
        points[:,K] = sb[:,K_large]
        
        min_d_ds = 2
        for i=1:K
            for j=i+1:K
                dist = norm(points[:,i]-points[:,j]);   
                if dist < min_d_ds
                    min_d_ds = dist; 
                end
            end
        end
    else        
        # surface area of a hypersphere, since we tile it with K hypercubes
        if N%2 == 1
            AN = N*2^N*pi^((N-1)/2)*factorial((N-1)/2)/factorial(N)
        else
            AN = N*pi^(N/2)/(factorial(N/2))
        end

        # approximate edge length for N-1 dimensional hypercube
        delta = big((AN/K)^(1/(N-1)))

        # k isn't exact, so we have to iteratively change delta until we get 
        #  the k we actually want
        true_k = 0
        while true_k < K
            points, true_k = place_on_sphere(delta, N, K, false)
            delta = big(delta*(true_k/K)^(1/(N-1)))
        end

        points, true_k = place_on_sphere(delta, N, K, true)
    end
    
    # use_sp means that one of the points is already given, so we need 
    #  to rotate the sphere to get them to coincide
    if use_sp == true
        points = rotate_points(points, sp, N, K)
    end
    
    return points
end

# iterative procedure to get a set of points nearly uniformly on 
#  the unit hypersphere. if use_sp flag is set, rotate to one of the points
function place_on_sphere(delta, N, K, actually_place)
    points = zeros(BigFloat, N, K)
    points_idx = 1
    idx = 1

    curr_angle = zeros(BigFloat, N-1, 1)
    generate_new_delta = true

    while idx<N && points_idx <= K    
        if generate_new_delta == true
            if idx == 1
                delt_idx = delta
            else
                delt_idx = big(delta/prod(sin.(curr_angle[1:idx-1])))
            end
        end

        if (idx < N-1 && curr_angle[idx] + delt_idx > big(pi)) || (idx == N-1 && curr_angle[idx] + delt_idx > 2*big(pi))
            if idx == 1
                # we're done with all the points we can produce
                break 
            else
                generate_new_delta = true
                idx = idx-1
                # reset the angle down:
                curr_angle[idx+1] = big(0)
            end
        else
            curr_angle[idx] = curr_angle[idx] + delt_idx
                
            if idx == N-1
                # we'll iterate through all the angles at the N-1 level
                generate_new_delta = false

                # generate point from spherical coordinates
                if actually_place
                    point = coord_from_angle(curr_angle, N)

                    # add the point to our list:
                    points[:,points_idx] = point
                end
                points_idx = points_idx+1
            else
                idx = idx+1
            end        
        end
    end
    
    true_k = points_idx-1
    return [points, true_k]
end


# place children. just performs the inversion and then uses the uniform
#  unit sphere function to actually get the locations
function add_children_dim(p, x, dim, edge_lengths, use_codes, SB, Gen_matrices; verbose=false)
    (p0,x0) = (reflect_at_zero(x,p), reflect_at_zero(x,x))
    c       = length(edge_lengths)
    q       = norm(p0)
        
    if verbose println("p = $(convert(Array{Float64,1}, p))") end    
    if verbose println("x = $(convert(Array{Float64,1}, x))") end    
    if verbose println("p0 = $(convert(Array{Float64,1}, p0))") end    
    if verbose println("x0 = $(convert(Array{Float64,1}, x0))") end    

    assert(norm(p0) <= 1.0)

    # a single child is a special case, place opposite the parent:
    if c == 1
        points0 = zeros(BigFloat, 2, dim);
        points0[2,:] = big(-1)*p0./norm(p0);
    else
        if use_codes
            points0 = place_children_codes(dim, c+1, true, p0./norm(p0), Gen_matrices)
        else
            points0 = place_children(dim, c+1, true, p0./norm(p0), true, SB)
        end
        points0 = points0'
    end
    
    points0[1,:] = p;
    for i=2:c+1
        points0[i,:] = reflect_at_zero(x, edge_lengths[i-1]*points0[i,:])
    end
     
    return points0[2:end,:]
end

# higher dimensional combinatorial hyperbolic embedding
function hyp_embedding_dim(G_BFS, root, eps, weighted, dim, tau, d_max, use_codes)    
	n             = G_BFS[:order]()
    T             = zeros(BigFloat, n, dim)
    
    root_children = collect(G_BFS[:successors](root));
    d             = length(root_children);

    edge_lengths  = hyp_to_euc_dist(tau*ones(d,1));

    # if the tree is weighted, need to set the edge lengths: 
    if weighted
        k = 1;
        for child in root_children
            weight = G[root+1][child+1]["weight"]
            edge_lengths[k] = hyp_to_euc_dist(weight*tau);
            k = k+1;
        end
    end
    
    v = Int(ceil(log2(d_max)))

    if use_codes
        # new way: generate a bunch of generator matrices we'll use for our codes 
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
    end
    
    if  !use_codes || d_max > dim
        SB_points = 1000
        SB        = place_children(dim, SB_points, false, 0, false, 0) 
    end

    
    # place the children of the root:
    if use_codes && d <= dim
        R = place_children_codes(dim, d, false, 0, Gen_matrices)
    else
        R = place_children(dim, d, false, 0, true, SB)
    end
    
    R = R'
    for i=1:d
        R[i,:] *= edge_lengths[i]
    end
    
    for i=1:d
         T[1+root_children[i],:] = R[i,:]
    end
    
    # queue containing the nodes whose children we're placing
    q = [];
    append!(q, root_children)
    node_idx = 0
    
    while length(q) > 0    
        h            = q[1];
        node_idx     += 1
        if node_idx%100 == 0
            println("Placing children of node $(node_idx)")
        end
        
        children     = collect(G_BFS[:successors](h));          
        parent       = collect(G_BFS[:predecessors](h));
        num_children = length(children);
        edge_lengths = hyp_to_euc_dist(tau*ones(num_children,1));

        append!(q, children)

        if weighted
            k = 1;
            for child in children
                weight = G[h+1][child+1]["weight"];
                edge_lengths[k] = hyp_to_euc_dist(big(weight)*tau);
                k = k+1;
            end
        end
        
        if num_children > 0
            if use_codes && num_children+1 <= dim
                R = add_children_dim(T[parent[1]+1,:], T[h+1,:], dim, edge_lengths, true, 0, Gen_matrices)
            else
                R = add_children_dim(T[parent[1]+1,:], T[h+1,:], dim, edge_lengths, false, SB, 0)
            end
            
            for i=1:num_children
                T[children[i]+1,:] = R[i,:];
            end        
        end    
        
        deleteat!(q, 1)    
    end

    return T
end