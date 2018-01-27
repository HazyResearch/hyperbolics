using Polynomials
using JLD

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
        v         /= norm(v[j+1:n])
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

tol=big(0.1)^(20)
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

# diagm(a) + diagm(b,1) + diagm(b,-1)
#
function solve_tridiagonal(a, b, y)
    n = length(a)
    # Keep the coefficients down. Do gaussian elimination
    # c_i b_i        c_i b_i
    # b_i a_{i+1}      0 c_{i+1}
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

function inverse_power_T(a,b,mu; T=100)
    n  = length(a)
    x = randn(n); x /= norm(x)
    y  = solve_tridiagonal(a-mu,b,x)  # replace with tri solve
    l  = dot(y,x)
    mu = mu + 1/l
    err = norm(y - l * x) / norm(y)
    for k=1:T
        x = y / norm(y)
        #y = (A - mu * I) \ x;
        y = solve_tridiagonal(a-mu,b,x)
        l = y' * x;
        mu = mu + 1 / l
        err = norm(y - l * x) / norm(y)
    end
    return mu,x
end

## Main function
function k_largest(A,k,tol)
    (n,n) = size(A)
    (T,U) = hess(A)
    println("\t Tridiagonal formed $(Float64(vecnorm(U*A*U' - T)))")
    f     = compute_char_poly(T)
    println("\t Char poly computed")

    roots = find_k_largest_roots(f,k,tol)
    println("\t Largest Roots found")

    # now we solve
    _eigs = zeros(BigFloat, n, k)
    ee    = zeros(BigFloat, k)
    for i=1:k
        (mu, v)    = inverse_power_T(diag(T), diag(T,1), roots[i])
        _eigs[:,i] = v
        ee[i]      = mu
    end
    ee, U'_eigs, T
end

function quick_serialize(data_set, fname, scale, prec)
    setprecision(BigFloat, prec)
    G = dp.load_graph(data_set)
    H = gh.build_distance(G, scale, num_workers=num_workers) 
    n,_ = size(H)
    Z = (cosh.(big.(H))-1)./2
    (T,U) = hess(A)
    println("\t Tridiagonal formed $(Float64(vecnorm(U*A*U' - T)))")
    save(fname, "T", T, "U", U)
end

function serialize(data_set, fname, scale, k_max, prec, num_workers=4)
    setprecision(BigFloat, prec)
    G = dp.load_graph(data_set)
    H = gh.build_distance(G, scale, num_workers=num_workers) 
    n,_ = size(H)
    Z = (cosh.(big.(H))-1)./2

    println("First e call")
    tic()
    eval, evec, T0 = k_largest(Z,1,tol)  
    lambda = eval[1]
    u      = evec[:,1]
    println("lambda = $(convert(Float64,lambda))")
    toc()
    
    u = u[1] < 0 ? -u : u

    b     = big(1) + sum(u)^2/(lambda*u'*u);
    alpha = b-sqrt(b^2-big(1));
    u_s   = u./(sum(u))*lambda*(big(1)-alpha);
    d     = (u_s+big(1))./(big(1)+alpha);
    dinv  = big(1)./d;
    v     = diagm(dinv)*(u_s.-alpha)./(big(1)+alpha);
    D     = big.(diagm(dinv));
    
    M = -(D * Z * D - ones(n) * v' - v * ones(n)')/2;
    M = (M + M')/2;
      
    tic()
    (T,U) = hess(M)
    println("\t Tridiagonal formed error=$(Float64(vecnorm(U*M*U' - T)))")
    toc()
    JLD.save(fname,"T",T,"U",U,"M",M)
    println("\t Saved into $(fname)")
end
