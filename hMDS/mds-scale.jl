using PyCall
using JLD
#using GenericSVD
@pyimport numpy as np
@pyimport networkx as nx
@pyimport scipy.sparse.csgraph as csg
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport data_prep as dp
@pyimport load_dist as ld
@pyimport distortions as dis

setprecision(BigFloat, 1024)

function big_gemv!(A,x_in,x_temp)
    (n,n) = size(A)
    Threads.@threads for i=1:n
        x_temp[i] = big(0.)
        for j=1:n
            x_temp[i] += A[i,j]*x_in[j]
        end
    end
    Threads.@threads for i=1:n
       x_in[i] = x_temp[i]
    end
end

function power_method(A,d,tol;T=1000)
    (n,n) = size(A)
    x_all = big.(qr(randn(n,d))[1])
    _eig  = zeros(BigFloat, d)
    x_temp = zeros(BigFloat,n)
    (nx,cur_dist) = (big(0.), big(0.))
    for j=1:d
        tic()	
        x = view(x_all,:,j)
        x /= norm(x)
        for t=1:T            
            #x = A*x
            big_gemv!(A,x,x_temp)
            if j > 1
                #x -= sum(x_all[:,1:(j-1)]*diagm(vec(x'x_all[:,1:(j-1)])),2)
                yy = vec(x'view(x_all, :,1:(j-1)))
                for k=1:(j-1)
                    x -= view(x_all,:,k)*yy[k]
                end
            end
            nx = norm(x)
            x /= nx
            cur_dist = abs(nx - _eig[j])
            if !isinf(cur_dist) &&  min(cur_dist, cur_dist/nx) < tol
                println("\t Done with eigenvalue $(j) val=$(Float64(nx)) at iteration $(t) at abs_tol=$(Float64(abs(nx - _eig[j]))) rel_tol=$(Float64(abs(nx - _eig[j])/nx)) tol=$(Float64(tol))")
                break
            end
            if t % 500 == 0
                println("\t $(t) $(Float64(cur_dist))\n\t\t $(Float64(cur_dist/nx))")
            end
            _eig[j]    = nx
        end
	println("\t $(j) val=$(Float64(_eig[j])) tol=$(Float64(tol)) abs=$(Float64(cur_dist)) rel=$(Float64(cur_dist/nx))")
        x_all[:,j] = x 
	toc()
    end
    return (_eig, x_all)
end

# Compute Z=A'A 
function matrix_square(A)
    Z = zeros(A)
    tic()
    println("Squaring...")
    Threads.@threads for i=1:n
        for k=1:n
            for j=1:n
                Z[i,k] += A[i,j]*A[k,j]
            end
        end
    end
    toc()
    return Z
end

function power_method_sign(A,r,tol;verbose=false, T=500)
    _d, _U    = power_method(matrix_square(A),r, tol;T=T)
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
    degrees = Dict(G[:degree]());
    cd      = collect(degrees);
    d_max   = maximum([cd[i][2] for i in 1:n])

    (nu, tau) = (0, 0)
    	
    beta    = big(pi)/(big(1.2)*d_max)
    v       = -2*k*log(tan(beta/2))
    
    for edge in G[:edges]()   
        (deg1, deg2) = (degrees[edge[1]], degrees[edge[2]])      
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
function center_inplace(A)
    (n,n) = size(A)
    mu    = vec(mean(A,1))
    Threads.@threads for i=1:n A[i,:] -= mu end

    mu = vec(mean(A,2))
    Threads.@threads for i=1:n A[:,i] -= mu end
end

# this is classical MDS
function mds(Z, k, n)
    #o = ones(n,1)
    #H = eye(n)-1/n*o*o'
    #B = -1/2*H*Z*H
    #B = 1/2*(B+B')
    println("Entering MDS $(k) $(n)")
    tic()
    Zc = -copy(Z)/2
    center_inplace(Zc)
    println("\t MDS Centering Complete")
    toc()
    tic()
    lambdasM, usM = power_method_sign(Zc,k,tol)
    println("Power Method Complete")
    toc()
    pos_idx = lambdasM .> 0.
    
    ## posE = 0
    ## while (posE < k && lambdasM[posE+1] > 0)
    ##     posE+=1;
    ## end

    Xrec = usM[:,pos_idx] * diagm(lambdasM[pos_idx] .^ 0.5);    
    return Xrec', sum(pos_idx)
end


# hMDS exact:
function h_mds(Z, k, n, tol)
    println("First e call $(tol) for k=$(k) n=$(n)")
    tic()
    eval, evec   = power_method_sign(Z,2,tol)  
    (lambda,idx) = findmax(eval)
    u            = evec[:,idx]
    println("lambda = $(convert(Float64,lambda)) idx=$(idx)")
    toc()
    
    if (u[1] < 0)
        u = -u
    end
    assert(lambda >= 0 && minimum(u) >= 0)
    b     = big(1.) + sum(u)^2/(lambda*dot(u,u))
    alpha = b-sqrt(max(b^2-big(1.),0))
    u_s   = u./(sum(u))*lambda*((1)-alpha)
    d     = (u_s+(1))./((1)+alpha)
    dinv  = (1)./d
    println("b=$(Float64(b)) alpha=$(Float64(alpha))")
    
    #v     = diagm(dinv)*(u_s.-alpha)./((1)+alpha);
    v     = dinv.*(u_s.-alpha)./((1)+alpha)
    #D     = diagm(dinv)
    Z     = copy(Z)
    Threads.@threads for i=1:n
        for j=1:n
            Z[i,j] *= dinv[i]*dinv[j]
        end
    end
    # Threads.@threads for i=1:n
    #     for j=1:n
    #         Z[i,j] -= (v[i] + v[j])
    #     end
    #end
    center_inplace(Z)
    Z/=(-2.0)
       
    # power method:
    println("Second e call $(tol)")
    tic()
    lambdasM, usM = power_method_sign(Z,k,tol) 
    pos_idx = lambdasM .> 0.
    println("\t $(Float64.(lambdasM[pos_idx]))")
    Xrec = usM[:,pos_idx] * diagm(lambdasM[pos_idx] .^ 0.5);
    Xrec = Xrec'
    toc()
    
    return Xrec, sum(pos_idx)
end

data_set = parse(Int32,(ARGS[1]))
k = parse(Int32, (ARGS[2]))
scale = parse(Float64, (ARGS[3]))
prec = parse(Int64, (ARGS[4]))
tol = parse(Float64, (ARGS[5]))

setprecision(BigFloat, prec)

#println("Scaling = $(convert(Float64,scale))");
#println(string("./dists/dist_mat",data_set,".p"))   

#H = ld.load_dist_mat(string("./dists/dist_mat",data_set,".p"));
tic()
G = dp.load_graph(data_set)
H = ld.get_dist_mat(G);
toc()
n,_ = size(H)
println("Graph Loaded with $(n) nodes tol=$(tol)")


Z = (cosh.(big.(H.*scale))-1)./2
println("Doing HMDS...")
tic()
Xrec, found_dimension = h_mds(Z, k, n, tol)
# save the recovered points:
save(string("Xrec_dataset_",data_set,"r=",k,"prec=",prec,"tol=",tol,".jld"), "Xrec", Xrec);
toc()

println("Building recovered graph... $(found_dimension)")
tic()
Zrec = big.(zeros(n, n))

Threads.@threads for i = 1:n
    for j = 1:n	 
        Zrec[i,j] = norm(Xrec[:,i] - Xrec[:,j])^2 / ((1 - norm(Xrec[:,i])^2) * (1 - norm(Xrec[:,j])^2));
    end
end 
println("min=$(Float64(minimum(Zrec)))")
toc()

println("Getting metrics")
tic()
Hrec = acosh.(1+2*Zrec)
Hrec = convert(Array{Float64,2},Hrec)
Hrec /= convert(Float64,scale)
 
println("----------------h-MDS Results-----------------")   
dist_max, dist, good = dis.distortion(H, Hrec, n, 2)
println("Distortion avg/max, bad = $(convert(Float64,dist)), $(convert(Float64,dist_max)), $(convert(Float64,good))")  
mapscore = dis.map_score(H, Hrec, n, 2) 
println("MAP = $(mapscore)")   
println("Dimension = $(found_dimension)")
toc() 

#####################3
## MDS
#
Xmds, dim_mds = mds(H, k, n)
tic()
# the MDS distances:
Zmds = zeros(n,n)
Threads.@threads for i = 1:n
    for j = 1:n
        Zmds[i,j] = norm(Xmds[:,i] - Xmds[:,j])
    end
end
toc()
    
    
 println("----------------MDS Results-----------------")
 dist_max, dist, bad = dis.distortion(H, Zmds, n, 2)
 println("MDS Distortion avg/max, bad = $(convert(Float64,dist)), $(convert(Float64,dist_max)), $(convert(Float64,bad))")  
 mapscore = dis.map_score(H, Zmds, n, 2)
 println("MAP = $(mapscore)")   
 println("Bad Dists = $(bad)")
 println("Dimension = $( dim_mds)") 
    
