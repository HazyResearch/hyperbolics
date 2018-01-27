using Polynomials

function deriv(f) return Poly([ k*coeffs(f)[k+1] for k=1:degree(f)]) end
function sign_pattern(F,x) return [Int64(sign((F[j](x)))) for j=1:length(F)] end
function sign_changes(F,x) return sum(abs.(diff(sign_pattern(F,x))) .> 1) end
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

function sturm_algorithm(f,a,b,tol)
    println("Starting $( (a,b) )")
    F  = build_sturm_sequence(f)
    sturm_binary_search(F, a, b, tol)
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
    ch = sign_changes(F,a) - sign_changes(F,b)
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
        while any(sign_pattern(F,mid) .== 0)
            mid            += abs(hi-lo)/big(500.0)
            jiggle_counter += 1
            assert(mid < hi && jiggle_counter < 100)
        end

        #if sign(g(a)*g(b)) < 1 || abs(a-b) < tol || abs(g(mid)) < tol return Set([(1,a,b)]) end
        if abs(lo-hi) < tol || abs(g(mid)) < tol return (1,mid) end
        lhs = sign_changes(F,lo  ) - sign_changes(F,mid)
        rhs = sign_changes(F,mid)  - sign_changes(F,hi)
        if !(xor(lhs > 0,rhs > 0))
            println("lhs=$(lhs) rhs=$(rhs)")
            
            println("a=$(lo)\n\t $(sign_changes(F,lo))")
            println("mid=$(mid)\n\t $(sign_changes(F,mid))")
            println("mid=$(hi)\n\t $(sign_changes(F,hi))")
            
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
    ch = sign_changes(F,a) - sign_changes(F,b)
    if ch == 1 return single_binary_search(F,a,b,tol) end

    # Pick the next mid point, jiggling if needed.
    mid = (a+b)/big(2.)
    while any(sign_pattern(F,mid) .== 0)
        mid += tol/2.0
        assert(mid < b)
    end

    #
    # This code checks for multiple roots
    # 
    if (abs(a-b) < tol || abs(g(mid)) < tol) return Set([(ch,mid)]) end

    # 
    lhs = sign_changes(F,a  ) - sign_changes(F,mid)
    rhs = sign_changes(F,mid) - sign_changes(F,b)

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
    g  = F[1]
    ch = sign_changes(F,a) - sign_changes(F,b)
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
            println("\t Found $(ch) root. Single. active=$(length(active)) completed=$(length(completed)) roots=$(active_roots())")
            continue
        end
        assert( ch > 0 )
        # Pick the next mid point, jiggling if needed.
        mid = (a+b)/big(2.)
        jiggle_counter = 0
        while any(sign_pattern(F,mid) .== 0)
            mid += tol/big(8.0)
            jiggle_counter += 1
            assert(mid < b && jiggle_counter < 1000)
        end

        #
        # This code checks for multiple roots
        # 
        #if (abs(a-b) < tol || abs(g(mid)) < tol)
        if (tol_check(a,b,tol) || abs(g(mid)) < tol)
            println("\t Found $(ch) root. Multiple. active=$(length(active)) completed=$(length(completed)) roots=$(active_roots())")
            push!(completed, (ch,mid))
            continue
        end

        # break the intervals in two
        lhs = sign_changes(F,a  ) - sign_changes(F,mid)
        rhs = sign_changes(F,mid) - sign_changes(F,b)

        assert( lhs + rhs == ch )
        if lhs > 0 push!(active, (lhs,a  ,mid) ) end
        if rhs > 0 push!(active, (rhs,mid,b  )  ) end
        if (lhs > 0) && (rhs > 0)
            println("\t Broken! $(ch) -> $(lhs)  $(rhs) ")
            println("\t\t active=$(length(active)) completed=$(length(completed)) roots=$(active_roots())")
            println("\t\t $([z[1] for z in active])")
        end
    end
    return completed
end


function sturm_search(F,a,b,tol)
    root_pairs = sturm_binary_search_queue(F,a,b,tol)
    (ms,rs)    = collect(zip(root_pairs...))
    (ms,rs)    = (collect(ms), collect(rs))
    idx        = sortperm(rs,rev=true)
    return (ms[idx], rs[idx])
end
