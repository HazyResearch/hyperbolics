using PyCall
#using GenericSVD
@pyimport numpy as np
@pyimport networkx as nx
@pyimport scipy.sparse.csgraph as csg
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport data_prep as dp
@pyimport load_dist as ld
#@pyimport distortions as dis

setprecision(BigFloat, 8192)


function power_method(A,d;T=5000, tol=big(1e-100))
    (n,n) = size(A)
    x_all = big.(qr(randn(n,d))[1])
    _eig  = zeros(BigFloat, d)
    for j=1:d
        x = x_all[:,j]
        x /= norm(x)
        for t=1:T
            x = A*x
            if j > 1
                x -= sum(x_all[:,1:(j-1)]*diagm(vec(x'x_all[:,1:(j-1)])),2)
            end
            nx = norm(x)
            x /= nx
            if abs(nx - _eig[j]) < tol
                println("\t Done with eigenvalue $(j) at iteration $(t) at $(Float64(abs(nx - _eig[j]))) ")
                break
            end
            _eig[j]    = nx
        end
        x_all[:,j] = x 
    end
    return (_eig, x_all)
end

function power_method_sign(A,r;verbose=false, T=5000)
    _d, _U    = power_method(A'A,r;T=T)
    X         = _U'A*_U 
    _d_signed = vec(diag(X))
    if verbose
        print("Log Off Diagonals: $( Float64(log(vecnorm( X - diagm(_d_signed)))))")
    end
    return _d_signed, _U
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
    
    for edge in G[:edges]()   
        (deg1, deg2) = (degrees[edge[1]+1], degrees[edge[2]+1])      
        alpha        = 2*big(pi)/(max(deg1,deg2))-2*beta
        len          = -big(2)*k*log(tan(alpha/2))
        w            = weighted ? edge[2]["weight"] : 1
        nu           = (len/w > nu) ? len/w : nu
        tau          = (1+eps)/eps*v > w*nu ? ((1+eps)/eps*v)/w : nu
    end
    return tau
end


function distortion(H1, H2)
    n,_ = size(H1)
    mc, me, avg, good = 0,0,0,0;
    for i=1:n
        for j=i+1:n
            if !isnan(H2[i,j]) && H2[i,j] != Inf && H2[i,j] != 0 && H1[i,j] != 0
                avg += max(H2[i,j]/H1[i,j], H1[i,j]/H2[i,j]);

                if H2[i,j]/H1[i,j] > me
                    me = H2[i,j]/H1[i,j]
                end

                if H1[i,j]/H2[i,j] > mc
                    mc = H1[i,j]/H2[i,j]
                end
                good += 1
            end
        end
    end

    avg/=(good);
    return (convert(Float64, mc*me), convert(Float64, avg), n*(n-1)/2-good)
end
# hMDS exact:
function h_mds(Z, k, n)
    println("First e call")
    tic()
    eval, evec = power_method_sign(Z,1)  
    lambda = eval[1]
    u = evec[:,1]

    toc()
    #EZ = eig(Z);
    #lambda = EZ[1][n];
    #u = EZ[2][:,n];    
    
    if (u[1] < 0)
        u = -u;
    end;

    b     = big(1) + sum(u)^2/(lambda*u'*u);
    alpha = b-sqrt(b^2-big(1));
    u_s   = u./(sum(u))*lambda*(big(1)-alpha);
    d     = (u_s+big(1))./(big(1)+alpha);
    dinv  = big(1)./d;
    v     = diagm(dinv)*(u_s.-alpha)./(big(1)+alpha);
    D     = big.(diagm(dinv));
    
    M = -(D * Z * D - ones(n) * v' - v * ones(n)')/2;
    M = (M + M')/2;
       
    # power method:
    println("Second e call")
    tic()
    lambdasM, usM = power_method_sign(M,k) 
    posE = 1;
    while (lambdasM[posE] > 0 && posE<k)
        posE+=1;
    end

    Xrec = usM[:,1:posE-1] * diagm(lambdasM[1:posE-1] .^ 0.5);
    Xrec = Xrec';
    toc()
    
    # low precision:
    #EM = eig(M);    
    #lambdasM = EM[1][(n-k+1):n];
    #usM = EM[2][:,(n-k+1):n];

    # using SVD:
    #sv = svdfact(M)
    #A = diagm(sv[:S].^0.5)
    #Xrec = (sv[:U][:,1:k]*A[1:k,1:k])'
       
    return Xrec, posE-1
end

data_set = parse(Int32,(ARGS[1]))
k = parse(Int32, (ARGS[2]))
eps = parse(Float64, (ARGS[3]))

G = dp.load_graph(data_set)

#println("Loaded graph on $(n) nodes");

# scale factor from combinatorial embedding
scale = get_emb_par(G, 1, eps, false)
println("Scaling = $(convert(Float64,scale))");
println(string("./dists/dist_mat",data_set,".p"))   

H = ld.load_dist_mat(string("./dists/dist_mat",data_set,".p"));
n,_ = size(H)

H *= scale;
Z = (cosh.(H)-1)./2

println("Doing HMDS...")
tic()
Xrec, found_dimension = h_mds(Z, k, n)
toc()

if found_dimension > 1
    println("Building recovered graph...")
    tic()
    Zrec = big.(zeros(n, n));
    for i = 1:n
        for j = 1:n
            Zrec[i,j] = norm(Xrec[:,i] - Xrec[:,j])^2 / ((1 - norm(Xrec[:,i])^2) * (1 - norm(Xrec[:,j])^2));
        end
    end
    toc()

    println("Getting distortion")
    tic()
    Hrec = acosh.(1+2*Zrec)
    dist_max, dist, good = distortion(H, Hrec)
    toc()
    println("Distortion avg/max, dimension = $(dist), $(dist_max), $(found_dimension)")  
else
    println("Dimension = 1!")
end

