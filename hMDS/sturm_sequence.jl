using Polynomials
#
# http://homepage.divms.uiowa.edu/~atkinson/m171.dir/sec_9-4.pdf
# http://www.maths.ed.ac.uk/~aar/papers/bamawi.pdf
#
function deriv(f) return Poly([ k*coeffs(f)[k+1] for k=1:degree(f)]) end
function mult_root(f,x,tol)
    g = deepcopy(f)
    for i=0:degree(f)
        if abs(g(x)) > tol return i end
        g = deriv(g)
    end
    return degree(f)
end
function sign_pattern(F,x, tol)
    big_enough = [j for j=1:length(F) if abs(F[j](x)) > tol]
    return [Int64(sign(F[j](x))) for j in big_enough]
end

function sign_changes(F,x,tol) return sum(abs.(diff(sign_pattern(F,x,tol))) .> 1) end
function build_sturm_sequence(f)
    n    = degree(f)+1
    F    = zeros(Poly{BigFloat}, n)
    F[1] = f
    F[2] = deriv(f)
    k    = 2
    for j=3:n
        k = j
        F[j] = -rem(F[j-2], F[j-1])
        if F[j] == 0
            break
        end
    end
    println("\t Chain of length $(k) of $(n)")
    return F[1:k]
end



# http://www.maths.ed.ac.uk/~aar/papers/bamawi.pdf
# q sequence
function sturm_seq(a,b,z)
    n    = length(a)
    q    = zeros(BigFloat, n)
    q[1] = a[1] - z
    for i=2:n
        q[i] = (a[i] - z) - b[i]^2/q[i-1] 
    end
    return Int64.(sign.(q))
end
#
# assumes [lo,hi] is a bracket i.e., sign(g(lo)*g(hi)) < 0
#
function local_bisection(g,lo,hi, tol; T=5000)
   mid = big(0.)
   for t = 1:T
       assert(sign(g(lo))*sign(g(hi)) < 0.0)
       mid = (lo+hi)/big(2.)
       if abs(lo-hi) < tol || abs(g(mid)) < tol return (1,mid) end
       if sign(g(mid)*g(hi)) < 0.
            lo = mid
       else
            hi = mid
       end
   end
   println("WARN: Bisection did not reach $(tol) at $(abs(g(mid))) $(T) distance = $(abs(hi-lo)) ")
   return (1,mid)
end

function single_binary_search(F,a,b,tol;T=1000)
    ch = sign_changes(F,a,tol) - sign_changes(F,b, tol)
    g  = F[1]
    assert(ch == 1)
    lo,hi  = a,b
    for t=1:T
        # check if we found a bracket, and use regular bisection then
        if g(lo)*g(hi) < 0.0
            println("\t\t Found a bracket. Starting bisection. ")
            return local_bisection(g,lo,hi, tol)
        end

        #
        # Find a good sturm midpoint.
        #
        mid = (lo+hi)/big(2.)
        jiggle_counter = 0
        while any(sign_pattern(F,mid,tol) .== 0)
            mid            += abs(hi-lo)/big(2.0)
            jiggle_counter += 1
            assert(mid < hi && jiggle_counter < 100)
        end

        #if sign(g(a)*g(b)) < 1 || abs(a-b) < tol || abs(g(mid)) < tol return Set([(1,a,b)]) end
        if abs(lo-hi) < tol || abs(g(mid)) < tol return (1,mid) end
        lhs = sign_changes(F,lo,tol )  - sign_changes(F,mid,tol)
        rhs = sign_changes(F,mid,tol)  - sign_changes(F,hi,tol)
        if !(xor(lhs > 0,rhs > 0))
            println("lhs=$(lhs) rhs=$(rhs) ch=$(ch)")
            
            println("a  =$(lo)\n\t $(sign_changes(F,lo,tol))")
            println("mid=$(mid)\n\t $(sign_changes(F,mid,tol))")
            println("hi =$(hi)\n\t $(sign_changes(F,hi,tol))")
            
            println(" $(abs(a-b))\n\t $(abs(g(mid)))\n $( min( abs(big(1.)-lo/hi), abs(big(1.) - lo/hi)) )") 
            assert(false)
        end
        if lhs > 0 
            hi = mid
        else
            lo = mid
        end
        if t % 100 == 0 print(".") end
    end
    println("Search without convergence")
    println("a=$(lo)\nb=$(hi)")
    mid = (hi+lo)/big(2.)
    println("$(abs(lo-hi))\n\t $(abs(g(mid)))\n $( min( abs(big(1.)-lo/hi), abs(big(1.) - hi/lo)) )") 
    assert(false)
end


function sturm_binary_search(F,a,b,tol)
    g  = F[1]
    ch = sign_changes(F,a,tol) - sign_changes(F,b,tol)
    if ch == 1 return single_binary_search(F,a,b,tol) end

    # Pick the next mid point, jiggling if needed.
    mid = (a+b)/big(2.)
    while any(sign_pattern(F,mid,tol) .== 0)
        mid += tol/2.0
        assert(mid < b)
    end

    #
    # This code checks for multiple roots
    # 
    if (abs(a-b) < tol || abs(g(mid)) < tol) return Set([(ch,mid)]) end

    # 
    lhs = sign_changes(F,a  ,tol) - sign_changes(F,mid,tol)
    rhs = sign_changes(F,mid,tol) - sign_changes(F,b,tol)

    r = Set()
    assert( lhs > 0 || rhs > 0)
    if lhs > 0 r = union(r, sturm_binary_search(F,a, mid, tol)) end
    if rhs > 0 r = union(r, sturm_binary_search(F,mid, b, tol)) end
    return r
end

function tol_check(a,b, tol)
    r = abs(a-b)
    return min( abs(r/a), abs(r/b), r ) < tol
end

function sturm_binary_search_queue(F,a,b,tol)
    g         = F[1]
    ch        = sign_changes(F,a,tol) - sign_changes(F,b,tol)
    active    = push!([], (ch,a,b))
    completed = []
    function active_roots() return length(active) > 0 ? sum([z[1] for z in active]) : 0 end
    println("Starting with $(ch) -- $(active_roots()) to find -- $(tol)")
    while length(active) > 0
        # Breadth first strategy
        (ch,a,b) = shift!(active) # pop!(active)
        if ch == 1
            println("\t Search started log_gap = $(Float64.(log(abs(b-a))))")
            (ch,r) = single_binary_search(F,a,b,tol)
            push!(completed, (ch,r))
            println("\t Found $(ch) root ~$(Float64(r)). Single. active=$(length(active)) completed=$(length(completed)) roots=$(active_roots())")
            continue
        end
        assert( ch > 0 )
        # Pick the next mid point, jiggling if needed.
        mid = (a+b)/big(2.)
        jiggle_counter = 0
        while any(sign_pattern(F,mid,tol) .== 0)
            mid += tol/big(8.0)
            jiggle_counter += 1
            assert(mid < b && jiggle_counter < 1000)
        end

        #
        # This code checks for multiple roots
        # 
        #if (abs(a-b) < tol || abs(g(mid)) < tol)
        if (tol_check(a,b,tol) || abs(g(mid)) < tol)
            println("\t Found $(ch) roots ~$(Float64(mid)). Multiple. active=$(length(active)) completed=$(length(completed)) roots=$(active_roots())")
            push!(completed, (ch,mid))
            continue
        end

        # break the intervals in two
        lhs = sign_changes(F,a  ,tol) - sign_changes(F,mid,tol)
        rhs = sign_changes(F,mid,tol) - sign_changes(F,b,tol)

        assert( lhs + rhs == ch )
        if lhs > 0 push!(active, (lhs,a  ,mid) ) end
        if rhs > 0 push!(active, (rhs,mid,b  )  ) end
        if (lhs > 0) && (rhs > 0)
            println("\t Broken! $(ch) -> $(lhs) + $(rhs) ~[$(Float64.([a,mid,b]))]")
            println("\t\t active=$(length(active)) completed=$(length(completed)) roots=$(active_roots())")
            println("\t\t $([z[1] for z in active])")
        end
    end
    return completed
end


function sturm_search(F,a,b,tol)
    root_pairs = sturm_binary_search_queue(F,a,b,tol)
    (ms,rs)    = collect(zip(root_pairs...))
    # Toss away these ms, as they mean something different!
    (ms,rs)    = (collect(ms), collect(rs))
    ms         = [max(ms[i],mult_root(F[1],rs[i],tol)) for i in 1:length(rs)]
    idx        = sortperm(rs,rev=true)
    return (ms[idx], rs[idx])
end
