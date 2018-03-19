# utilities.jl
# various functions needed for combinatorial embeddings


# Reflection (circle inversion of x through orthogonal circle centered at a)
function isometric_transform(a, x)
    # r   = sqrt(norm(a)^2 - big(1.))
    # return (r/norm(x - a))^2*(x-a) + a
    r2 = norm(a)^2 - big(1.)
    return r2/norm(x - a)^2 * (x-a) + a
end

# Inversion taking mu to origin
function reflect_at_zero(mu,x)
    a = mu/norm(mu)^2
    return isometric_transform(a,x)
end

# Express a hyperbolic distance in the unit disk
function hyp_to_euc_dist(x)
    return sqrt.((cosh.(x)-big(1))./(cosh.(x)+big(1)))
    # return (exp.(x)-big(1))./(exp.(x)+big(1))
end

# Place children
function add_children(p,x,edge_lengths; verbose=false)
    (p0,x0) = (reflect_at_zero(x,p),reflect_at_zero(x,x))
    c       = length(edge_lengths);
    q       = norm(p0)
    p_angle = acos(p0[1]/q)
    if p0[2] < 0
        p_angle = 2*big(pi)-p_angle
    end
    alpha   = 2*big(pi)/(big(c+1.))

    if verbose println("p = $(convert(Array{Float64,1}, p))") end
    if verbose println("x = $(convert(Array{Float64,1}, x))") end
    if verbose println("p0 = $(convert(Array{Float64,1}, p0))") end
    if verbose println("x0 = $(convert(Array{Float64,1}, x0))") end
    assert(norm(p0) <= 1.0)

    points0 = zeros(BigFloat, c+1, 2)
    for k=1:c
        angle          = p_angle + alpha*k
        points0[k+1,1] = edge_lengths[k]*cos( angle )
        points0[k+1,2] = edge_lengths[k]*sin( angle )
    end
    for k=0:c
        points0[k+1,:] = reflect_at_zero(x,points0[k+1,:])
    end
    return points0[2:end,:]
end

# Get scaling factor tau
function get_emb_par(G, k, eps, weighted)
    n       = G[:order]();
    degrees = G[:degree]();
    cd      = collect(degrees);
	d_max   = maximum([cd[i][2] for i in 1:n])

    (nu, tau) = (0, 0)

    beta    = big(pi)/(big(1.2)*d_max)
    v       = -2*k*log(tan(beta/2))
    m       = length(G[:edges])
    idx     = 1

    if weighted
        # minimum weight edge:
        w = Inf
        for edge in G[:edges](data=true)
            ew = edge[3]["weight"]
            w  = ew < w ? ew : w
        end
        if w == Inf
            w = 1
        end
    else
        w = 1
    end

    _, d_max     = gu.max_degree(G)
    alpha        = 2*big(pi)/(d_max)-2*beta
    len          = -big(2)*k*log(tan(alpha/2))
    nu           = (len/w > nu) ? len/w : nu
    tau          = (1+eps)/eps*v > w*nu ? ((1+eps)/eps*v)/w : nu

    return tau
end

# Perform a combinatorial embedding into hyperbolic disk
# Construction based on Sarkar, "Low Distortion Delaunay
# Embedding of Trees in Hyperbolic Plane"
#  G_BFS should be a directed, rooted tree, in order to
#  use the networkx functions
function hyp_embedding(G_BFS, root, eps, weighted, tau)
    n             = G_BFS[:order]()
    T             = zeros(BigFloat,n,2)

    root_children = collect(G_BFS[:successors](root));
    d             = length(root_children);

    edge_lengths  = hyp_to_euc_dist(tau*ones(d,1));

    # if the tree is weighted, need to set the edge lengths:
    if weighted
        k = 1;
        for child in root_children
            weight = G_BFS[root+1][child+1]["weight"]
            edge_lengths[k] = hyp_to_euc_dist(weight*tau);
            k = k+1;
        end
    end

    # place the children of the root:
    for i=1:d
         T[1+root_children[i],:] = edge_lengths[i]*[cos(2*(i-1)*big(pi)/BigFloat(d)),sin(2*(i-1)*big(pi)/BigFloat(d))]
    end

    # queue containing the nodes whose children we're placing
    q = [];
    append!(q, root_children)

    while length(q) > 0
        h            = q[1];
        children     = collect(G_BFS[:successors](h));
        parent       = collect(G_BFS[:predecessors](h));
        num_children = length(children);
        edge_lengths = hyp_to_euc_dist(tau*ones(num_children,1));

        append!(q, children)

        if weighted
            k = 1;
            for child in children
                weight = G_BFS[h+1][child+1]["weight"];
                edge_lengths[k] = hyp_to_euc_dist(big(weight)*tau);
                k = k+1;
            end
        end

        if num_children > 0
            R = add_children(T[parent[1]+1,:], T[h+1,:], edge_lengths)
            for i=1:num_children
                T[children[i]+1,:] = R[i,:];
            end
        end

        deleteat!(q, 1)
    end

    return T
end


# Parallel version
function hyp_embedding_parallel(G_BFS, root, eps, weighted, edges_weights, tau)
	n             = G_BFS[:order]()
    T             = zeros(BigFloat,n,2)

    root_children = collect(G_BFS[:successors](root));
    d             = length(root_children);

    edge_lengths  = hyp_to_euc_dist(tau*ones(d,1));

    # if the tree is weighted, need to set the edge lengths:
    if weighted
        k = 1;
        for child in root_children
            weight = G_BFS[root+1][child+1]["weight"]
            edge_lengths[k] = hyp_to_euc_dist(weight*tau);
            k = k+1;
        end
    end

    #println("Placing node $(root)")

    # place the children of the root:
    for i=1:d
         T[1+root_children[i],:] = edge_lengths[i]*[cos(2*(i-1)*big(pi)/BigFloat(d)),sin(2*(i-1)*big(pi)/BigFloat(d))]
         #println("Placed child $(root_children[i])")
         #println(convert(Array{Float64,1},T[root_children[i]+1,:]))
    end

    # queue containing the nodes whose children we're placing
    q = [];
    append!(q, root_children)

    q_lock    = SpinLock()
    _cond     = Condition()
    _not_done = true
    function not_done()
        u = true
        lock(q_lock) do
            u = _not_done
        end
        return u
    end
    function grab_job()
        u = nothing
        lock(q_lock) do
            if length(q) > 0
                u = pop!(q)
            end
        end
        return u
    end
    everythread() do
        while not_done()
            h = grab_job()
            if h  == nothing
                wait(_cond) # wait to be woken up
                continue
            end
            h            = pop!(q)
            #println("Placing children of node $(h)")
            children     = collect(G_BFS[:successors](h))
            lock(q_lock) do
                append!(q, children)
            end

            parent       = collect(G_BFS[:predecessors](h));
            num_children = length(children);
            edge_lengths = hyp_to_euc_dist(tau*ones(num_children,1));

            if weighted
                k = 1;
                for child in children
                    weight = G_BFS[h+1][child+1]["weight"];
                    edge_lengths[k] = hyp_to_euc_dist(big(weight)*tau);
                    k = k+1;
                end
            end
        end

        if num_children > 0
            R = add_children(T[parent[1]+1,:], T[h+1,:], edge_lengths)
            for i=1:num_children
                #println("Placed child $(children[i])")
                T[children[i]+1,:] = R[i,:];
                #println(convert(Array{Float64,1},T[children[i]+1,:]))
            end
        end

    end

    return T
end
