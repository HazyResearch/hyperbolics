using Polynomials
using JLD
module Sturm include("sturm_sequence.jl") end

#
# Python
#
using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport data_prep as dp
@pyimport pytorch.graph_helpers as gh

function reflector(x)
    e = zeros(BigFloat,length(x))
    e[1] = 1
    u  = sign(x[1])*norm(x)*e + x
    u /= norm(u)
    return u
end

# U*_A*U' = A
function hess(_A)
    A     = copy(_A)
    (n,n) = size(A)
    U     = big.(eye(n))
    v     = zeros(BigFloat,n)
    
    for j = 1:(n-1)
        
        #u = reflector(vec(A[j+1:n,j]))
        # Note this is I - 2 * u *u '
        #v   = vcat(zeros(BigFloat,j), u)
        x          = A[j+1:n,j]
        #v[1:j]     = 0.
        v[j]       = 0.
        v[j+1:n]   = x
        v[j+1]    += sign(A[j+1,j])*norm(x)
        nv         = norm(v[j+1:n])
        v         /= nv > big(0.) ? nv : big(1.) 
        u          = v[j+1:n]
        
        U  -= 2*v*(v'U) # (I-2*v*v')*U
        #
        #A[(j+1):n,j:n] = A[j+1:n,j:n] - 2*u*(u'*A[j+1:n,j:n])
        A[(j+1):n,j:n] -= 2*u*(u'*A[j+1:n,j:n])
        #A[j:n,(j+1):n] = A[j:n,j+1:n] - 2*(A[j:n,j+1:n]*u)*u'
        A[j:n,(j+1):n] -= 2*(A[j:n,j+1:n]*u)*u'
        #
        # These will be zero, but this just cleans them up
        #
        #A[(j+2):n,j]   = big(0.)
        #A[j,(j+2):n]   = big(0.) 
        if j % 16 == 0 print(".") end 
    end
    return A,U
end

function compute_char_poly(Z)
    (n,n) = size(Z)
    x     = poly([big(0.0)])
    fm1   = x - Z[1,1] 
    f     = (x-Z[2,2])*fm1 - Z[1,2]^2
    for i=3:n
        (f,fm1) = (x-Z[i,i])*f - Z[i-1,i]^2*fm1,f
    end
    return f
end


#
# This section is about computing the largest 0s
#
#
function knuth_bound(g)
    return 2*maximum([abs(coeffs(g)[degree(g)+1-k])^(big(1./k)) for k=1:degree(g)])
end
function root_square(g,x)
    evens  = sum([coeffs(g)[j]*x^(Int64((j-1)/2)) for j=1:2:(1+degree(g))])
    odds   = sum([coeffs(g)[j]*x^(Int64((j-2)/2)) for j=2:2:(1+degree(g))])
    g2 = evens^2 - x*odds^2
    return g2
end
#http://www.numdam.org/article/M2AN_1990__24_6_693_0.pdf
function solve_to_tol(g, tol)
    g_0 = deepcopy(g)
    x   = poly([big(0.0)])
    # tol > (2*degree(g))^(2^(-k)) - 1
    k = -Int64(ceil(log2(log(big(1.)+tol)/log(big(2*degree(g))))))
    for j=1:k
        g_0 = root_square(g_0,x)
    end
    z = big(2.)^(-k)
    return knuth_bound(g_0)^(z),k
end

#
# assumes [lo,hi] is a bracket i.e., sign(g(lo)*g(hi)) < 0
#
function bisection(g,lo,hi; tol=big(0.1)^20, T=5000)
   mid = big(0.)
   for t = 1:T
       assert(sign(g(lo))*sign(g(hi)) < 0.0)
       mid = lo + (hi - lo)/big(2.)
       if abs(g(mid)) < tol return mid end
       if g(mid)*g(hi) < big(0.)
            lo = mid
       else
            hi = mid
       end
   end
   println("WARN: Bisection did not reach $(tol) at $(abs(g(mid))) $(T) distance = $(abs(hi-lo)) ")
   return mid
end

x  = poly([big(0.0)])
function deriv(f) return Poly([ k*coeffs(f)[k+1] for k=1:degree(f)]) end
function newton(f, u; tol=tol, T=5000)
    df = deriv(f)
    x  = copy(u)
    for t=1:T
        x_next = x - f(x)/df(x)
        if abs(x_next - x)/x_next <= tol && f(x_next) <= tol
            #println("Done in $(t) iterations $(f(x_next))")
            return x_next
        end
        x = x_next
    end
    println("WARNING: Newton failed to converge")
    return x
end

function find_largest_root(g, tol;use_bisection=false)
    s_tol = big(0.1)^(12) # this is our root resolution!
    u,k   = solve_to_tol(g, s_tol)
    z     = big(2.)^(-float(k))
    m_err = big(2.*degree(g))^(z)
    #println("\t\t $(sign(g(u/m_err))) $(sign(g(u))) $(u)")
    z     = use_bisection ? bisection(g, u/m_err, u; tol=tol) : newton(g, u, tol=tol)
        
    return z,div(g, x-z)
end

function find_k_largest_roots(g, k, tol)
    gg = deepcopy(g)
    roots = zeros(BigFloat, k)
    for i=1:k
        (u,gg) = find_largest_root(gg, tol)
        roots[i] = u
    end
    return roots
end

# 
function solve_tridiagonal_tol(a, b, y, tol)
    return SymTridiagonal(a,b)\y
end
    ## n = length(a)
    ## # Keep the coefficients down. Do gaussian elimination
    ## # c_i b_i     |   c_i b_i
    ## # b_i a_{i+1} |     0 c_{i+1}
    ## # c_{i+1} = a_{i+1} - b_i^2/c_i
    ## c    = zeros(BigFloat,n)
    ## d    = zeros(BigFloat,n)
    ## c[1] = a[1]
    ## d[1] = y[1]
    ## row_ops = []
    ## for i=2:n
    ##     if abs(c[i-1]) > tol
    ##         c[i] = a[i]-b[i-1]^2/c[i-1] 
    ##         d[i] = y[i]-(b[i-1]*d[i-1])/c[i-1]
    ##     elseif abs(b[i-1]) < tol
    ##         c[i-1] = 0.0
    ##     else
    ##         # This is a column operation, which we now have to keep track of
    ##         # 0   b_i     -b_i             b_i
    ##         # b_i c_i -->  b_i - c_{i+1}   c_{i+1}
    ##         # namely \hat{x}_{i} = x_i - x_{i+1}
    ##         c[i-1] = -b[i-1]
    ##         c[i]   = a[i] + (b[i-1]-c[i-1])
    ##         d[i]   = y[i] + (b[i-1]-c[i-1])*d[i-1]/b[i-1]
    ##     end
    ## end
    
    ## # Now do back substitution
    ## x    = zeros(BigFloat, n)
    ## if abs(c[n]) < tol
    ##     if abs(d[n]) >= tol return None end
    ##     #if abs(d[n]) < tol
    ##     assert(abs(d[n]) < tol)
    ##     x[n] = big(0.0) # both are zero, so unneeded.
    ## else
    ##     x[n] = d[n]/c[n]
    ## end
    
    ## for j=(n-1):-1:1
    ##     # if c is zero then d = b*x[j+1] + c*x[j]
    ##     if abs(c[j]) < tol
    ##         if abs(b[j]*x[j+1] - d[j]) >= tol return None
    ##         println("$(Float64(b[j]*x[j+1] - d[j])) b=$(Float64(b[j])) x=$(Float64(x[j+1])) d=$(Float64(d[j])) c=$(Float64(c[j]))")
    ##         #assert()
    ##     else
    ##         x[j] = (d[j] - b[j]*x[j+1])/c[j]
    ##     end
    ## end
    ##return x
##end


# diagm(a) + diagm(b,1) + diagm(b,-1)
#
function solve_tridiagonal(a, b, y)
    n = length(a)
    # Keep the coefficients down. Do gaussian elimination
    # c_i b_i     |   c_i b_i
    # b_i a_{i+1} |     0 c_{i+1}
    # c_{i+1} = a_{i+1} - b_i^2/c_i
    c    = zeros(BigFloat,n)
    d    = zeros(BigFloat,n)
    c[1] = a[1]
    d[1] = y[1]
    for i=2:n
        c[i] = a[i]-b[i-1]^2/c[i-1] 
        d[i] = y[i]-(b[i-1]*d[i-1])/c[i-1]
    end
    
    # Now do back substitution
    x    = zeros(BigFloat, n)
    x[n] = d[n]/c[n] 
    for j=(n-1):-1:1
        x[j] = (d[j] - b[j]*x[j+1])/c[j]
    end
    return x
end

function tri_multiply(a,b,x)
    if length(size(x)) == 1
        n         = length(x)
        r         = a .* x
        r[1:n-1] += b .* x[2:n]
        r[2:n]   += b .* x[1:n-1]
        return r
    end

    # matrix version
    (n,d) = size(x)
    r     = zeros(x)
    for i=1:d
        r[:,i]      = a .* x[:,i]
        r[1:n-1,i] += b .* x[2:n,i]
        r[2:n,i]   += b .* x[1:n-1,i]
    end
    return r
end

function inverse_power_T_single(a,b,mu, tol; T=128)
    n  = length(a)
    x = tri_multiply(a,b,randn(n)); x /= norm(x)
    
    y  = solve_tridiagonal_tol(a-mu,b,x, tol)  # replace with tri solve
    l  = dot(y,x)
    mu = mu + 1/l
    err = norm(y - l * x) / norm(y)
    for k=1:T
        x  = y / norm(y)
        y  = solve_tridiagonal_tol(a-mu,b,x,tol)
        l  = y' * x;
        mu = mu + big(1.) / l
        err = norm(y - l * x) / norm(y)
        if err < tol
            println("\t\t\t Inverse_power_T_sngle Early Exit $(k)")
            break
        end
        if k % 16 == 0 print(".") end 
    end
    println("\t\t Single Eigenvector Differences $(Float64(log(vecnorm(tri_multiply(a,b,x) - x*mu)))) $(Float64(err)) ~$(Float64(mu))")
    return mu,x
end

# Multiple roots.
function s_proj(x,tol)
    z           = sqrt.(vec(mapslices(sum, abs2.(x), 1)))
    zero_idx    = abs.(z) .< tol
    z           = 1./z
    z[zero_idx] = big(0.0)
    return x*diagm(z)
end

function o_proj(x) qr(x)[1] end

# Really a inverse rayleigh iteration
function inverse_power_T(a,b,mu, m, tol; T=100, o_tol=big(1e-8))
    if m == 1 
        _mu, _x = inverse_power_T_single(a,b, mu, tol)
        return (reshape([_mu], 1), reshape(_x,length(_x),1))
    end
    
    
    n       = length(a)
    x       = o_proj(big.(randn(n,m)))
    y       = zeros(x)
    mus     = big.(copy(mu))
    println("\t\t\t--> $(Float64.(mus))")
    err     = big.(zeros(m))
    function step()
        for i=1:m
            assert(!any(isnan.(x)))
            u = solve_tridiagonal_tol(a-mus[i],b,x[:,i], tol)  # replace with tri solve
            # Primal Errors
            ## yy      = y[:,i]/norm(y[:,i])
            ## z       = tri_multiply(a,b,yy)
            ## mu_t    = dot(z,yy)
            ## err[i]  = abs(mu_t-mus[i])
            ## mus[i]  = mu_t

            # Dual Errors
            l       = dot(y[:,i],x[:,i])
            mus[i]  = abs(l) > tol ? mus[i] + big(1.)/l : mus[i]
            #err[i]  = abs(l) > tol ? norm(y[:,i] - l * x[:,i]) / norm(y[:,i]) : big(0.0)
            err[i]  = abs(norm(y[:,i])) > tol ? norm(y[:,i] - l * x[:,i]) / norm(y[:,i]) : big(0.0) 
        end
    end
    step()
        
    # NB: Reorthogonalize?
    for k=1:T
        x = s_proj(y, tol)
        # Reorthogonalize, periodically.
        if vecnorm(I - x'x) > o_tol x = o_proj(x) end
        step()
        if maximum(err) < tol 
            raw_error   = Float64(log(vecnorm(tri_multiply(a,b,x) - x*diagm(mus))))
            ortho_error = Float64(log(vecnorm(I-x'x)))
            println("\t\t Eigenvector Differences $(m) iteration=$(k) raw=$(raw_error) ortho=$(ortho_error) $(Float64(maximum(err)))")
            return (mus,x)
        end
        if k % 10 == 0
            raw_error   = Float64(log(vecnorm(tri_multiply(a,b,x) - x*diagm(mus))))
            ortho_error = Float64(log(vecnorm(I-x'x)))
            println("\t\t <intermediate> Eigenvector Differences $(m) iteration=$(k) $(raw_error) ortho=$(ortho_error) $(Float64(maximum(err)))")
        end
    end
    raw_error   = Float64(log(vecnorm(tri_multiply(a,b,x) - x*diagm(mus))))
    ortho_error = Float64(log(vecnorm(I-x'x)))

    println("\t\t Eigenvector Differences ~$(Float64.(mus)) $(raw_error) ortho=$(ortho_error) $(Float64(maximum(err)))")
    return mus,x
end


function largest_sturm(f,a,b, tol) return Sturm.sturm_search(Sturm.build_sturm_sequence(f), a, b, tol) end



function k_largest_simple(A,_k,tol;use_blocks=true)
    # HACK!
    if use_blocks return k_largest_simple_block(A,_k,tol) end
    (n,n)   = size(A)
    (T,U)   = hess(A)
    matrix_blocks = collect(1:(n-1))[diag(T,1) .< tol]
    println("\t Tridiagonal formed $(Float64(vecnorm(U*A*U' - T)))")
    println("\t Number of small off-diagonal elements=$( sum(diag(T,1) .< tol))")
    println("\t Matrix Blocks = $(matrix_blocks)")
    
    hi      = maximum([maximum([T[i,i] + (abs(T[i,i-1]) + abs(T[i,i+1])) for i=2:(n-1)]), T[1,1] + abs(T[1,2]), T[n,n] + abs(T[n,n-1])])
    k       = min(n,_k)
    elems   = k == 1 ? maximum(hi) : (sort(diag(T),rev=true)[1:k])
    (mu, v) = inverse_power_T(diag(T), diag(T,1), elems , k, tol)
    idx     = sortperm(abs.(mu),rev=true)
    _vals   = mu[idx]
    _eigs   = v[:,idx]
    eig_vecs = U'_eigs

    A_apx_error = Float64(log(vecnorm(eig_vecs*diagm(_vals)*eig_vecs' - A)/vecnorm(A)))
    T_apx_error = Float64(log(vecnorm(_eigs*diagm(_vals)*_eigs' - T)/vecnorm(T)))
    println("\t Completed Eigenvectors. Norm Difference A_apx=$(A_apx_error) T_apx=$(T_apx_error) log(|A|)=$(Float64(log(vecnorm(A)))) log(|T|)=$(Float64(log(vecnorm(T))))")
    println("\t\t $(Float64(vecnorm(A))) $(Float64(vecnorm(T)))")
  
    return (_vals, eig_vecs, T)
end
    

## Main function
function k_largest(A,_k,tol)
    (n,n) = size(A)
    (T,U) = hess(A)
    println("\t Tridiagonal formed $(Float64(vecnorm(U*A*U' - T)))")
    f     = compute_char_poly(T)
    hi    = maximum([maximum([T[i,i] + (abs(T[i,i-1]) + abs(T[i,i+1])) for i=2:(n-1)]), T[1,1] + abs(T[1,2]), T[n,n] + abs(T[n,n-1])])
    lo    = minimum([minimum([T[i,i] - (abs(T[i,i-1]) + abs(T[i,i+1])) for i=2:(n-1)]), T[1,1] - abs(T[1,2]), T[n,n] - abs(T[n,n-1])])
    
    println("\t Char poly computed lo=$(lo) hi=$(hi)")
    (m,roots) = largest_sturm(f, lo, hi, tol)
    println("\t All Roots found $(length(roots)) $(sum(m))")

    #
    # Solve for eigenvectors
    #
    n_distinct_roots = length(roots)
    n_roots          = min(sum(m),_k)

    _eigs = zeros(BigFloat, n, n_roots)
    _vals = zeros(BigFloat, n_roots)
    cur   = 1
    for i=1:n_distinct_roots
        _m              = m[i]
        (mu, v)         = inverse_power_T(diag(T), diag(T,1), roots[i]*ones(_m), _m, tol)
        println("\t\t\t Returned eigenvector for values ~$(Float64.(mu)) of size $(size(v)) with multiplicity $(_m) reported as $(m[i])")
        new_m           = Sturm.mult_root(f,mu[1],tol)
        println("\t\t\t After polish m = $(new_m) $(Float64.(vecnorm(mu-mu[1])))")
        # TODO: IF new $m$ is higher, rerun! (and check if the vector goes up--and orthogonal)
        if new_m > _m
            println("\t\t\t RERUN!")
            (new_mu, new_v) = inverse_power_T(diag(T), diag(T,1), roots[i]*ones(new_m), new_m, tol)
            good_idx        = abs.(new_mu - mu[1]) .<= tol
            if sum(good_idx) > _m
                mu = new_mu[good_idx]
                v  = new_v[:,good_idx]
                _m = sum(good_idx)
                println("\t\t\t Rerun yieled $(_m) vectors instead")
            end
        end
        # Keep it in bounds
        _e              = min(n_roots,cur+_m-1)
        _to_keep        = _e - cur + 1
        println("\t\t\t Indexing info: cur=$(cur) _e=$(_e) $(n_roots) $(_to_keep)")
        _eigs[:,cur:_e] = v[:,1:_to_keep]
        _vals[cur:_e]   = mu[1:_to_keep]
        
        cur += _m
        if _e >= n_roots break end
    end
    eig_vecs = U'_eigs

    A_apx_error = Float64(log(vecnorm(eig_vecs*diagm(_vals)*eig_vecs' - A)/vecnorm(A)))
    T_apx_error = Float64(log(vecnorm(_eigs*diagm(_vals)*_eigs' - T)/vecnorm(T)))
    println("\t Completed Eigenvectors. Norm Difference A_apx=$(A_apx_error) T_apx=$(T_apx_error) log(|A|)=$(Float64(log(vecnorm(A)))) log(|T|)=$(Float64(log(vecnorm(T))))")
    println("\t\t $(Float64(vecnorm(A))) $(Float64(vecnorm(T)))")
    # HACK. REMOVE
    a_eigs=eigs(Float64.(A))[1]
    t_eigs=eigs(Float64.(T))[1]
    println("\t\t HACK REMOVE ME $(a_eigs) $(length(a_eigs)) $(rank(Float64.(A))) $(size(A))")
    println("\t\t t_eigs=$(t_eigs)")
    println("\t\t ee    =$(Float64.(_vals))")
    return (_vals, eig_vecs, T)
end

## function quick_serialize(data_set, fname, scale, prec)
##     setprecision(BigFloat, prec)
##     G = dp.load_graph(data_set)
##     H = gh.build_distance(G, scale, num_workers=num_workers) 
##     n,_ = size(H)
##     Z = (cosh.(big.(H))-1)./2
##     (T,U) = hess(A)
##     println("\t Tridiagonal formed $(Float64(vecnorm(U*A*U' - T)))")
##     save(fname, "T", T, "U", U)
## end

function compute_d(u,l,n, scale_hack=true)
    assert( minimum(u) >= big(0.) )
    b       = big(1.) + sum(u)^2/(l*norm(u)^2)
    
    alpha = b - sqrt(b^2-1.)
    v   = u*(l*(1.-alpha))/sum(u)
    d   = (v+1.)/(1.+alpha)
    d_min = minimum(d)
    if d_min < 1
        println("\t\t\t Warning: Noisy d_min correction used.")
        d/=d_min
    end
    dv  = d - 1 
    return (d,dv)
end

function load_graph(dataset; num_workers=1)
    G = dp.load_graph(dataset)
    return gh.build_distance(G, 1.0, num_workers=num_workers)
end

function serialize(dataset, fname, scale, k_max, prec, num_workers=4, simple=true)
    setprecision(BigFloat, prec)
    tol = big(2.)^(-400)
    println("Prec=$(prec) log_tol=$(Float64(log2(tol)))")
    H = load_graph(dataset, num_workers=num_workers)
    n,_ = size(H)
    Z = (cosh.(scale*big.(H))-1)./big(2.)
    
    println("First pass. All Eigenvectors. rt_err=$(vecnorm(Float64.(acosh.(1+2*Z)/scale)-H)) $(any(isnan.(Z)))")
    tic()
    eval, evec, T = simple ? k_largest_simple(Z, 1, tol) : k_largest(Z,1,tol)  
    lambda = eval[1]
    u      = evec[:,1]
    println("lambda = $(convert(Float64,lambda)) test=$(Float64(vecnorm(Z*u - lambda*u)))")
    toc()

    # u is the Perron vector, so it should be the same sign.
    # Wlog, we consider it to be negative.
    u = u[1] < 0 ? -u : u
    println("Perron = $(Float64.(minimum(u)))")

    d,dv  = compute_d(u,lambda,n)
    println("DEBUG")
    println("D1=$(Float64.(d))\n")
    println("Dv=$(Float64.(dv))\n")

    inv_d = big(1)./d
    D     = diagm(d);
    v     = big(1) - inv_d;
    # TODO: Do this faster!
    Q     = (I-ones(BigFloat, n,n)/big(n))*diagm(inv_d)
    G  = -Q*Z*Q'/2
    ## M0 = -(diagm(inv_d) * Z * diagm(inv_d) - big.(ones(n)) * v' - v * big.(ones(n))')/2;
    ## M0 = (M0 + M0')/big(2);   
    ## println("DIFFERENCES = $(Float64(vecnorm(M0 - G)))")
    M  = G
    println("Gram matrix constructed. Creating dataset.")
    tic()
    M_val, M_eigs, M_T = simple ? k_largest_simple(M,k_max, tol) : k_largest(M,k_max,tol)
    toc()
    
    
    JLD.save(fname,"T",T,"M",M,"M_val", M_val, "M_eigs", M_eigs, "M_T", M_T, "H", H, "prec", prec, "dataset", dataset, "scale", scale )
    println("\t Saved into $(fname)")
    return (H,M_val, M_eigs, M_T)
end


## New block-based factorization
function op_factor(T,_k,tol)
    (n,n)   = size(T)
    assert(  n >= 2 )
    
    hi      = max(T[1,1] + abs(T[1,2]), T[n,n] + abs(T[n,n-1]))
    if n > 2
        hi      = max(maximum([T[i,i] + (abs(T[i,i-1]) + abs(T[i,i+1])) for i=2:(n-1)]),hi)
    end
    k       = min(n,_k)
    elems   = k == 1 ? maximum(hi) : (sort(diag(T),rev=true)[1:k])
    (mu, v) = inverse_power_T(diag(T), diag(T,1), elems, k, tol)
    idx     = sortperm(abs.(mu),rev=true)
    _vals   = mu[idx]
    _eigs   = v[:,idx]
    return _vals, _eigs
end

function k_largest_simple_block(A,_k,tol)
    (n,n)   = size(A)

    tic()
    (T,U)   = hess(A)
    matrix_blocks = collect(1:(n-1))[abs.(diag(T,1)) .< tol]
    println("\t Tridiagonal formed $(Float64(vecnorm(U*A*U' - T)))")
    println("\t Number of small off-diagonal elements=$( sum(abs.(diag(T,1)) .< tol))")
    println("\t <blocks> Matrix Blocks = $(matrix_blocks) $(any(isnan.(A))) $(any(isnan.(T)))")
    toc()

    _eigs  = zeros(T)
    _vals  = zeros(BigFloat, n)
    _start = 1
    for _end in matrix_blocks
        idx = _start:_end
        #
        # handle 1x1 and 2x2 blocks in special cases
        #
        if _start == _end
            _vals[_start]        = T[_start,_start]
            _eigs[_start,_start] = big(1.)
        elseif _start + 1 == _end
            _T = T[idx,idx]
            tr = trace(_T)
            _det = _T[1,1]*_T[2,2] - _T[2,1]^2
            _vals[_start]  = tr/2 + sqrt(tr^2/4 - _det)
            _vals[_end  ]  = tr/2 - sqrt(tr^2/4 - _det)
            if abs(_T[1,2]) < tol
                _eigs[_start,_start] = 1
                _eigs[_end  ,_end]   = 1
            else
                _eigs[_start,_start] = _vals[_start] - _T[2,2]
                _eigs[_start,_end]   = _vals[_end  ] - _T[2,2]
                _eigs[_end, idx   ]  = _T[1,2]
            end
        else
            k                  = min(_k,length(idx))
            _op_vals, _op_eigs = op_factor(T[idx,idx], k, tol)
            _idx               = _start:(_start+k-1)
            _vals[_idx]        = _op_vals
            _eigs[idx,_idx]    = _op_eigs
        end
        _start = _end + 1
    end
    
    eig_vecs = U'_eigs
    A_apx_error = Float64(log(vecnorm(eig_vecs*diagm(_vals)*eig_vecs' - A)/vecnorm(A)))
    T_apx_error = Float64(log(vecnorm(_eigs*diagm(_vals)*_eigs' - T)/vecnorm(T)))

    println("\t Completed Eigenvectors. Norm Difference A_apx=$(A_apx_error) T_apx=$(T_apx_error) log(|A|)=$(Float64(log(vecnorm(A)))) log(|T|)=$(Float64(log(vecnorm(T))))")
    println("\t\t $(Float64(vecnorm(A))) $(Float64(vecnorm(T)))")


    # Note that if k < block size, we ee will not contain that vector
    # TODO: Wrap this in a validation section!
    # Also, use symmetric tridiagonal
    a_eigs=eigs(Float64.(A))[1]
    t_eigs=eigs(SymTridiagonal(Float64.(diag(T)), Float64.(diag(T,1))))[1]
    s_idx =sortperm(Float64.(abs.(_vals)), rev=true)
    println("\t\t Low-Precision Eigenvalues $(repr(a_eigs)) $(length(a_eigs)) $(rank(Float64.(A))) $(size(A))")
    println("\t\t t_eigs=$(t_eigs) $(length(t_eigs))")
    println("\t\t ee    =$(Float64.(_vals[s_idx])))")
    return (_vals, eig_vecs, T)
end
