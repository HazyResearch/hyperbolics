using Polynomials

function deriv(f) return Poly([ k*coeffs(f)[k+1] for k=1:degree(f)]) end
function sign_pattern(F,x) return [sign(Float64.(F[j](x))) for j=1:length(F)] end
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
    println("\t Chain of length $(k)")
    return F[1:k]
end

function sturm_algorithm(f,a,b,tol)
    println("Starting $( (a,b) )")
    F  = build_sturm_sequence(f)
    sturm_binary_search(F, a, b, tol)
end

function sturm_binary_search(F,a,b,tol)
    ch = sign_changes(F,a) - sign_changes(F,b)
    if ch == 1
        g = F[1]
        println("Found a root. Is a bracket? $(sign(g(a)*g(b)) < 1)")
        return Set([(a,b)])
    end
    mid = (a+b)/big(2.)
    while any(sign_pattern(F,mid) .== 0)
        mid += tol/2.0
    end
    lhs = sign_changes(F,a  ) - sign_changes(F,mid)
    rhs = sign_changes(F,mid) - sign_changes(F,b)

    r = Set()
    if lhs > 0 r = union(r, sturm_binary_search(F,a, mid, tol)) end
    if rhs > 0 r = union(r, sturm_binary_search(F,mid, b, tol)) end
    println("\t ... $(mid) $(lhs) $(rhs)")
    return r
end
