using HDF5, LinearAlgebra, SparseArrays,Dates,Random,ProgressBars, ITensors, ITensorMPS
include("H_building.jl")
include("Pauli Generator.jl")
include("Measurement_functions.jl")

#include("pauli_strings.jl")


function dimerization_from_b(b::AbstractVector)
    if b[1]==0
        b[1]=1
    elseif b[1]==1
    else
        b=vcat(1,b)
    end

    b_even = b[2:2:end]
    b_odd=b[3:2:end]
    m = min(length(b_even), length(b_odd))
    b_dimerization = b_odd[1:m] .- b_even[1:m]
    return b_dimerization
end


function dimerized_powerlaw_decay_bn(K_dim::Int,a::Real,b::Real,alpha::Real)
    bn = ones(K_dim-1) .* a
    for n in collect(2:2:K_dim-1)
        bn[n] += (b)*(n^(-alpha))
    end
    return bn
end

function Spectral_fct_from_bn(bn::Vector)
    L = L_matrix(bn)
    vals,vecs = eigen(L)

    return (vals, 2pi .* vecs[1,:] .^2)
end

L_sparse(bn) = spdiagm(-1 => bn, 1 => (bn))
L_matrix(bn) = Tridiagonal(bn,zeros(length(bn)+1),bn)

function K_phi_t(bn::Vector, t_vec::Vector,i::Int=1)
    # Diagonalize A
    
    x0=zeros(length(bn)+1); x0[1]=1 
    L = Tridiagonal(bn,zeros(length(bn)+1),(-1 .* bn))
    F = eigen(L)
    V = F.vectors      # columns are eigenvectors
    eig_vals = (F.values)  # eigenvalues

    # Transform initial condition
    y0 = V \ x0  # y0 = V^{-1} * x0
    xt=Float64[]
    # Solve in eigenbasis then transform back
    for t in ProgressBar(t_vec)
        push!(xt,real((V * (exp.(eig_vals .* t) .* y0)))[i]) 
    end
    
    return xt
end

function K_complexity(bn::Vector,t_vec::Vector)
end



function Lanczos(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64)


    #Base vector
    O = []
    b = Float64[1]
    error_0 = Float64[]
    error_1 = Float64[]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O,Probe)
    LO_0 = L_n(O[1],H,2)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    
    #Define O1
    push!(O,LO_0/b[2])
    
    for n in ProgressBar(3:Nsteps)
        A_n = L_n(O[2],H,n) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        push!(b,b_n)
        O[1]=O[2]
        O[2] = (A_n/b_n)
        #println("n=$n, bn=$b_n")
        #push!(error_0,Op_Inner(Probe,O[2]))
        #push!(error_1,Op_Inner(LO_0/b[2],O[2]))
        #@show n,error_0, error_1
    end
    return b
end

function Lanczos_ED(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64)

    H = Matrix(H)
    vals,vecs=eigen(H)
    Probe=vecs' * Matrix(Probe) * vecs


    #Base vector
    O = []
    b = Float64[1]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O,Probe)
    LO_0 = L_n_ED(O[1],vals)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    
    #Define O1
    push!(O,LO_0/b[2])
    error_0=Float64[]
    error_1=Float64[]

    for n in ProgressBar(3:Nsteps)
        A_n = L_n_ED(O[2],vals) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        push!(b,b_n)
        O[1]=O[2]
        O[2] = (A_n/b_n)
        #println("n=$n, bn=$b_n")
        push!(error_1, Op_Inner(LO_0/b[2],O[2]))
        push!(error_0,Op_Inner(Probe,O[2]))
    end
    return b
end
function Lanczos_diag(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64)


    #Base vector
    O = []
    b = Float64[0]
    d=Float64[0]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O,Probe)
    LO_0 = L_n(O[1],H,2)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    push!(d,sum(diag(LO_0)))
    
    #Define O1
    push!(O,LO_0/b[2])
    
    for n in 3:Nsteps
        A_n = L_n(O[2],H,n) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        push!(b,b_n)
        push!(d,sum(diag(A_n)))
        O[1]=O[2]
        O[2] = (A_n/b_n)
        #println("n=$n, bn=$b_n")
    end
    return b,d
end

function Lanczos_subtract(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H_1::Union{Matrix,DenseMatrix,SparseMatrixCSC},H_2::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64)


    #Base vector
    O_1 = [];O_2 = [];O_sb = []
    b_1 = Float64[0];b_2 = Float64[0];b_sb = Float64[0]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O_1,Probe),push!(O_2,Probe),push!(O_sb,Probe)
    LO_1_0 = L_n(O_1[1],H_1,2)
    LO_2_0 = L_n(O_2[1],H_2,2)
    LO_sb_0 = LO_1_0-LO_2_0
    #Define b1, b0 is set to 0
    push!(b_1,Op_Norm(LO_1_0))
    push!(b_2,Op_Norm(LO_2_0))
    push!(b_sb,Op_Norm(LO_sb_0))
    
    #Define O1
    push!(O_1,LO_1_0/b_1[2])
    push!(O_2,LO_2_0/b_2[2])
    push!(O_sb,LO_sb_0/b_sb[2])
    
    for n in ProgressBar(3:Nsteps)
        A_1_n = L_n(O_1[2],H_1,n) - b_1[n-1]*O_1[1]
        A_2_n = L_n(O_2[2],H_2,n) - b_2[n-1]*O_2[1]
        A_sb_n = A_1_n - A_2_n


        b1 = Op_Norm(A_1_n)
        b2 = Op_Norm(A_2_n)
        bsb = Op_Norm(A_sb_n)
        push!(b_1,b1);push!(b_2,b2);push!(b_sb,bsb)
        O_1[1]=O_1[2];O_2[1]=O_2[2];O_sb[1]=O_sb[2]
        O_1[2] = (A_1_n/b1);O_2[2] = (A_2_n/b2);O_sb[2] = (A_sb_n/bsb);
        #println("n=$n, bn=$b_n")
    end
    return b_1[2:end],b_2[2:end],b_sb[2:end]
end

function Lanczos_monic(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64)


    #Base vector
    O = []
    Δ = Float64[0]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O,Probe)
    LO_0 = L_n(O[1],H,2)
    #Define b1, b0 is set to 0
    push!(Δ,Op_Norm(LO_0)^2)
    
    #Define O1
    push!(O,LO_0)
    
    for n in 3:Nsteps
        A_n = L_n(O[2],H,n) - Δ[n-1]*O[1]
        Δ_n = (Op_Norm(A_n)^2)/(Op_Norm(O[2])^2)
        push!(Δ,Δ_n)
        O[1]=O[2]
        O[2] = (A_n)
        #println("n=$n, bn=$b_n")
    end
    return Δ[2:end]
end


function Op_Norm(O::MPO) #norm function for MPOs
    L_tot=length(O)
    return norm(O)/sqrt(2^L_tot)
end

function Op_Norm(O::Matrix) #norm function for dense matrices
    return sqrt(tr(O'*O)/size(O)[1])
end

function Op_Norm(O::SparseMatrixCSC) #norm function for sparse matrices
    return sqrt(sum(abs.(findnz(O)[3]).^2)/size(O)[1])
end

function Op_Inner(A,B)
    return tr(A'*B)/size(A)[1]
end

function L_n(O::SparseMatrixCSC,H::SparseMatrixCSC,n::Int64) #Liouvillian function for sparse matrices
    return H*O-O*H
end
function L_n_ED(Op::Matrix,Vals::Vector)
    return (Vals .- Vals') .* Op
end
function L_n(O::Matrix,H::Matrix,n::Int64) #Liouvillian function for dense matrices
    prod = H*O
    return prod + ((-1)^(n+1))*prod'
end

function L_n(O::Matrix,H::SparseMatrixCSC,n::Int64) #Liouvillian function for dense O, sparse H (recommended)
    prod = O*H
    prod += ((-1)^(n+1))*prod'
    return -(prod)
end


function L_n(O::MPO,H::MPO,n::Int64;kwargs...) #Liouvillian function for MPOs
    HO=apply(H,O;kwargs...)
    return +(HO,((-1)^(n+1))*dag(swapprime(HO,0,1)))
end



function Lanczos(Probe::MPO,H::MPO,Nsteps::Int64,s;kwargs...) #Lanczos for Itensor. kwargs would include cutoff or maxdim

    L_tot=length(H)

    Probe /= Op_Norm(Probe) 

    #Start Lanczos Algorithm
    Id = MPO(s,"Id")
    #Base vector
    O = []
    b = Float64[0]
    global runtimes=Float64[]
    global linkdims_arr=zeros(Nsteps+1,L_tot-1)

    #Define O0
    push!(O,apply(Probe,Id;kwargs...))
    linkdims_arr[1,:] = linkdims(O[1])

    ta = Dates.now()
    
    LO_0 = L_n(O[1],H,2;kwargs...)
    
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    tb = Dates.now(); runtime=((tb-ta).value)*0.001
    global  runtimes = push!(runtimes,runtime)

    #Define O1
    push!(O,apply(LO_0,Id;kwargs...)/b[2])
    global  linkdims_arr[2,:] = linkdims(O[2])
    

    @time for n in ProgressBar(3:Nsteps+1)
        ta = Dates.now()

        L0 = L_n(O[2],H,n;kwargs...)
        b0 = b[n-1]*O[1]
        A_n = -(L0, b0)
        
        #A_n = L_n(O[2],H,n;kwargs...)-b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        
        
        O_n = (apply(A_n,Id;kwargs...)/b_n)
        O[1] = O[2]
        O[2] = O_n

        tb = Dates.now(); runtime=((tb-ta).value)*0.001
        push!(b,b_n)
        global  linkdims_arr[n,:] = linkdims(O_n)
        global  runtimes = push!(runtimes,runtime)
        
    end
    return vcat(1,b[2:end]),runtimes
end  

function Lanczos_full(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64) #Lanczos for Itensor. kwargs would include cutoff or maxdim
    L = size(H)[1]
   #Base vector
   O = zeros(Nsteps+1,L,L)
   b = Float64[0]
   q = Float64[0]
   d = Float64[0]
   #Define O0
   Probe/=Op_Norm(Probe)
   O[1,:,:] = Probe
   LO_0 = L_n(O[1,:,:],H,2)
   #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    push!(q,Op_Norm(LO_0-diagm(diag(LO_0))))
    push!(d,sum(diag(LO_0)))
   
   #Define O1
   O[2,:,:]=LO_0/b[2]
   
   for n in ProgressBar(3:Nsteps+1)
       A_n = L_n(O[n-1,:,:],H,n) - b[n-1]*O[n-2,:,:]
       b_n = Op_Norm(A_n)
       q_n = Op_Norm(A_n-diagm(diag(A_n)))
       d_n = sum(diag(A_n))
       push!(b,b_n)
       push!(q,q_n)
       push!(d,d_n)
       O[n,:,:] = (A_n/b_n)
       #println("n=$n, bn=$b_n")
   end
   return b[2:end],q[2:end],d[2:end],O
end  

function x_from_b(bn)
    x_is = Float64[]
    for i in 1:1:length(bn[1:2:end])-1
        x_i = -log(bn[2i-1]/bn[2i])
        push!(x_is,x_i)
    end
    return x_is
end


var_of_sum(cov_M, N::Int64) = [sum(cov_M[1:n, 1:n]) for n in 1:N]

function x_generate(bn_tot,K)
    arr_length = length((bn_tot[:,1])[1:2:end])-1
    xs = zeros(arr_length,K)
    if K>size(bn_tot)[2]
        println("K larger than bn_tot[2]")
    else
        for k in 1:1:K
            @show k
            bn = bn_tot[1:end,k]
            x_is = Float64[]
            for i in 1:1:arr_length
                x_i = -log(bn[2i-1]/bn[2i])
                push!(x_is,x_i)
            end
                xs[:,k] = x_is
        end
    end
    return xs
end
function dimerization_statistics(value_vec)
    cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//Anderson_krylov//Outs//Lanczos_exports/")
    dimers_all = Vector{Float64}[]

    for (H_flag,L,W,Nsteps,Nsamples) in value_vec
        fid = h5open(replace(H_flag*"-L$L-W$W-Nsteps$Nsteps-Nsamples$Nsamples-Exported", "." => "_")*".h5","r")        
        G = fid["b"]
        bn_tot = read(G,"bn_tot")
        close(fid)
        dimers=zeros(Int(Nsteps/2)-1,Nsamples)
        for i in 1:length(Nsamples)
            bn = vcat(1,bn_tot[:,i])
            bn_even = bn[2:2:end-1]
            bn_odd = bn[3:2:end]
            dimers[:,i] = bn_odd .- bn_even
        end
        dimers_avg = mean(dimers,dims=2)[:]
    push!(dimers_all,dimers_avg)
    end
    return dimers_all
end


function bn_statistics(value_vec)
    cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//Anderson_krylov//Outs//Lanczos_exports/")
    bn_avgs=[]
    xss=[]
    xs_vars=[]
    xs_var_cumsums=[]
    xs_means=[]
    xs_medians=[]
    xs_covs=[]
    gn_covs=[]
    xs_mean_cumsums=[]
    xs_median_cumsums=[]
    for (H_flag,L,W,Nsteps,Nsamples) in value_vec
            fid = h5open(replace(H_flag*"-L$L-W$W-Nsteps$Nsteps-Nsamples$Nsamples-Exported", "." => "_")*".h5","r")        
            G = fid["b"]
            bn_tot = read(G,"bn_tot")
            bn_avg = read(G,"bn_avg")
        close(fid)

        #xs[iteration, realizations]
        #xs = x_generate(bn_tot,Nsamples)
        xs = x_generate(bn_tot,Nsamples)
        ratios = exp.(-1 .* xs)

        xs_cumsum = cumsum(xs,dims=1)


        xs_var = var(xs,dims=2)[:]
        xs_var_cumsum = var(xs_cumsum,dims=2)[:]
        xs_mean = mean(xs,dims=2)[:]
        xs_med = median(xs,dims=2)[:]
        xs_cov = cov(xs,dims=2)
        gn_cov = cov(xs_cumsum,dims=2)
        ratios = exp.(-1 .* xs)

        push!(bn_avgs,bn_avg)
        push!(xss,xs)
        push!(xs_vars,xs_var)
        push!(xs_var_cumsums,xs_var_cumsum)
        push!(xs_means,xs_mean)
        push!(xs_medians,xs_med)
        push!(xs_covs,xs_cov)
        push!(gn_covs,gn_cov)
        push!(xs_median_cumsums,median(xs_cumsum,dims=2)[:])
        push!(xs_mean_cumsums,mean(xs_cumsum,dims=2)[:])
    end
    return bn_avgs,xss,xs_vars,xs_var_cumsums,xs_means,xs_covs,gn_covs,xs_mean_cumsums, xs_medians, xs_median_cumsums
end
function bn_statistics_biased(value_vec)
    cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//Anderson_krylov//Outs//Lanczos_exports/")
    bn_avgs=[]
    xss=[]
    xs_vars=[]
    xs_var_cumsums=[]
    xs_means=[]
    xs_medians=[]
    xs_covs=[]
    gn_covs=[]
    xs_mean_cumsums=[]
    xs_median_cumsums=[]
    for (H_flag,L,W,a,Nsteps,Nsamples) in value_vec
            fid = h5open(replace(H_flag*"-L$L-W$W-a$a-Nsteps$Nsteps-Nsamples$Nsamples-Exported", "." => "_")*".h5","r")        
            G = fid["b"]
            bn_tot = read(G,"bn_tot")
            bn_avg = read(G,"bn_avg")
        close(fid)

        #xs[iteration, realizations]
        #xs = x_generate(bn_tot,Nsamples)
        xs = x_generate(bn_tot,Nsamples)
        ratios = exp.(-1 .* xs)

        xs_cumsum = cumsum(xs,dims=1)


        xs_var = var(xs,dims=2)[:]
        xs_var_cumsum = var(xs_cumsum,dims=2)[:]
        xs_mean = mean(xs,dims=2)[:]
        xs_med = median(xs,dims=2)[:]
        xs_cov = cov(xs,dims=2)
        gn_cov = cov(xs_cumsum,dims=2)
        ratios = exp.(-1 .* xs)

        push!(bn_avgs,bn_avg)
        push!(xss,xs)
        push!(xs_vars,xs_var)
        push!(xs_var_cumsums,xs_var_cumsum)
        push!(xs_means,xs_mean)
        push!(xs_medians,xs_med)
        push!(xs_covs,xs_cov)
        push!(gn_covs,gn_cov)
        push!(xs_median_cumsums,median(xs_cumsum,dims=2)[:])
        push!(xs_mean_cumsums,mean(xs_cumsum,dims=2)[:])
    end
    return bn_avgs,xss,xs_vars,xs_var_cumsums,xs_means,xs_covs,gn_covs,xs_mean_cumsums, xs_medians, xs_median_cumsums
end
function bn_statistics_MBL(value_vec)
    cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//UOH//Outs//ExportedBD2000")
    bn_avgs=[]
    xss=[]
    xs_vars=[]
    xs_var_cumsums=[]
    xs_means=[]
    xs_covs=[]
    xs_mean_cumsums=[]
    for (H_flag,L,W,site,Nsamples,BD) in value_vec
        fid = h5open(replace("bn_tot_W$(W)_L$(L)_Lrr0_site$(site)-d1-IT-BD$BD", "." => "_")*".h5","r")
            G = fid["b"]
            bn_tot = read(G,"bn_tot")[2:end,:]
            bn_avg = read(G,"bn_avg")[2:end]
        close(fid)

        #xs[iteration, realizations]
        #xs = x_generate(bn_tot,Nsamples)
        xs = x_generate(bn_tot,Nsamples)
        ratios = exp.(-1 .* xs)

        xs_cumsum = cumsum(xs,dims=1)


        xs_var = var(xs,dims=2)[:]
        xs_var_cumsum = var(xs_cumsum,dims=2)[:]
        xs_mean = mean(xs,dims=2)[:]
        xs_cov = cov(xs,dims=2)
        ratios = exp.(-1 .* xs)

        push!(bn_avgs,bn_avg)
        push!(xss,xs)
        push!(xs_vars,xs_var)
        push!(xs_var_cumsums,xs_var_cumsum)
        push!(xs_means,xs_mean)
        push!(xs_covs,xs_cov)
        push!(xs_mean_cumsums,mean(xs_cumsum,dims=2)[:])
    end
    return bn_avgs,xss,xs_vars,xs_var_cumsums,xs_means,xs_covs,xs_mean_cumsums
end

function Lanczos_cn_calc(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64)

    Probe = Matrix(Probe)
    Probe/=Op_Norm(Probe)
    H = Matrix(H)
    N=size(Probe)[1]
    N_2 = Int(round(N/2))
    println("calculating eigenvectors...")
    vecs=eigvecs(H)

    #Find eigenvalue that has the maximum overlap with the initial probe
    println("calculating overlaps...")
    N_overlaps = N_2  # how many top overlaps you want to keep

    all_overlaps = Float64[]
    all_indices = Int[]
    all_vecs = Vector{Vector{Float64}}()

    # Step 1: collect everything
    for i in 1:N
        vec = vecs[:, i]
        P_vec = vec * transpose(vec)
        overlap = Op_Inner(P_vec, Probe)
        push!(all_overlaps, overlap)
        push!(all_indices, i)
        push!(all_vecs, vec)
    end

    # Step 2: get indices of top N_overlaps overlaps
    top_idx = partialsortperm(all_overlaps, 1:N_overlaps,rev=true)
    #top_idx = randperm(length(all_overlaps))[1:N_overlaps]
    # Step 3: extract top N_overlaps results
    max_overlap   = all_overlaps[top_idx]
    max_overlap_i = all_indices[top_idx]
    overlap_vecs  = all_vecs[top_idx]

    vecs=nothing

    cn=complex(zeros(length(overlap_vecs),Nsteps+1))
    sums=zeros(3,Nsteps+1)
    cn_test=complex(zeros(2,Nsteps+1))
    errors = zeros(2,Nsteps+1)

    test_vec1 = zeros(N);test_vec1[1]=1
    test_vec2 = ones(N);
    test_vecs=[test_vec1,test_vec2]
    #Base vector
    println("starting Lanczos...")
    O = Vector{Matrix{Float64}}(undef, 2)
    O[1] = Probe
    b = Float64[1]
    ZM_real_space = zeros(N,N)
    #=
        sum_c = 0.0;sum_l=0.;sum_r=0.
    for n in collect(-10:1:10)
            for m in collect(-10:1:10)
                sum_c += (O[1])[N_2 + n, N_2 + m]
                sum_r += (O[1])[N_2-20 + n, N_2-20 + m]
                sum_l += (O[1])[N_2+20 + n, N_2+20 + m]
            end
        end
    sums[:,1] = [sum_c,sum_l,sum_r]
    =#
    #Define O0
    push!(O,Probe)
    cn[:,1]=cn_calc(overlap_vecs,O[1])
    cn_test[:,1]=cn_calc(test_vecs,O[1])
    ZM_real_space+=LIOM_from_b(b)[1]*O[1]

    LO_0 = L_n(O[1],H,2)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    
    #Define O1
    O[2] = LO_0 / b[2]
    cn[:,2]=cn_calc(overlap_vecs,O[2])
    cn_test[:,2]=cn_calc(test_vecs,O[2])
    ZM_real_space+=LIOM_from_b(b)[2]*O[2]
    
    #=
    sum_c = 0.0;sum_l=0.;sum_r=0.
    for n in collect(-10:1:10)
            for m in collect(-10:1:10)
                sum_c += (O[2])[N_2 + n, N_2 + m]
                sum_r += (O[2])[N_2-20 + n, N_2-20 + m]
                sum_l += (O[2])[N_2+20 + n, N_2+20 + m]
            end
    end
    sums[:,2] = [sum_c,sum_l,sum_r]
    =#

    for n in ProgressBar(3:Nsteps+1)
        A_n = L_n(O[2],H,n) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        O[1], O[2] = O[2], A_n/b_n
        push!(b,b_n)
        cn_test[:,n] = cn_calc(test_vecs,O[2])#./(1im^(n-1))
        cn[:,n]=cn_calc(overlap_vecs,O[2])#./(1im^(n-1))
        
        ZM_real_space+=LIOM_from_b(b)[n]*O[2]
        #=
        sum_c = 0.0;sum_l=0.;sum_r=0.
        for n in collect(-10:1:10)
            for m in collect(-10:1:10)
                sum_c += (O[2])[N_2 + n, N_2 + m]
                sum_r += (O[2])[N_2-20 + n, N_2-20 + m]
                sum_l += (O[2])[N_2+20 + n, N_2+20 + m]
            end
        end
        sums[:,n] = [sum_c,sum_l,sum_r]
        =#
        error_0 = abs(Op_Inner(Probe,O[2]))
        error_1 = abs(Op_Inner(LO_0/b[2],O[2]))
        errors[:,n] = [error_0,error_1]
        if error_0 > 1E-4 || error_1 > 1E-4
            @show n,error_0, error_1
        end
    end


    return b,cn,ZM_real_space,overlap_vecs,max_overlap,errors
end

function Lanczos_cn_calc_ED(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64)
    
    H = Matrix(H)
    vals,e_vecs=eigen(H)
    Probe=e_vecs' * Matrix(Probe) * e_vecs

    Probe/=Op_Norm(Probe)
    N=size(Probe)[1]
    N_2 = Int(round(N/2))
    println("calculating eigenvectors...")

    #Find eigenvalue that has the maximum overlap with the initial probe
    println("calculating overlaps...")
    N_overlaps = N_2  # how many top overlaps you want to keep

    all_overlaps = Float64[]
    all_indices = Int[]
    all_vecs = Vector{Vector{Float64}}()

    # Step 1: collect everything
    for i in 1:N
        vec = e_vecs[:, i]
        P_vec = vec * transpose(vec)
        overlap = Op_Inner(P_vec, Probe)
        push!(all_overlaps, overlap)
        push!(all_indices, i)
        push!(all_vecs, vec)
    end

    # Step 2: get indices of top N_overlaps overlaps
    top_idx = partialsortperm(all_overlaps, 1:N_overlaps,rev=true)
    #top_idx = randperm(length(all_overlaps))[1:N_overlaps]
    # Step 3: extract top N_overlaps results
    max_overlap   = all_overlaps[top_idx]
    max_overlap_i = all_indices[top_idx]
    overlap_vecs  = all_vecs[top_idx]

    vecs=nothing

    cn=complex(zeros(length(overlap_vecs),Nsteps+1))
    sums=zeros(3,Nsteps+1)
    cn_test=complex(zeros(2,Nsteps+1))
    errors = zeros(2,Nsteps+1)

    test_vec1 = zeros(N);test_vec1[1]=1
    test_vec2 = ones(N);
    test_vecs=[test_vec1,test_vec2]
    #Base vector
    println("starting Lanczos...")
    O = Vector{Matrix{Float64}}(undef, 2)
    O[1] = Probe
    b = Float64[1]
    ZM_real_space = zeros(N,N)


    #Define O0
    push!(O,Probe)
    cn[:,1]=cn_calc(overlap_vecs,O[1])
    cn_test[:,1]=cn_calc(test_vecs,O[1])
    ZM_real_space+=LIOM_from_b(b)[1]*(e_vecs *O[1]* e_vecs')

    LO_0 = L_n_ED(O[1],vals)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    
    #Define O1
    O[2] = LO_0 / b[2]
    cn[:,2]=cn_calc(overlap_vecs,O[2])
    cn_test[:,2]=cn_calc(test_vecs,O[2])
    ZM_real_space+=LIOM_from_b(b)[2]*(e_vecs *O[2]* e_vecs')
    


    for n in ProgressBar(3:Nsteps+1)
        A_n = L_n_ED(O[2],vals) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        O[1], O[2] = O[2], A_n/b_n
        push!(b,b_n)

        cn_test[:,n] = cn_calc(test_vecs,O[2])#./(1im^(n-1))
        cn[:,n]=cn_calc(overlap_vecs,O[2])#./(1im^(n-1))
        
        ZM_real_space+=LIOM_from_b(b)[n]*(e_vecs *O[2]* e_vecs')

        error_0 = abs(Op_Inner(Probe,O[2]))
        error_1 = abs(Op_Inner(LO_0/b[2],O[2]))
        errors[:,n] = [error_0,error_1]
        if error_0 > 1E-4 || error_1 > 1E-4
            @show n,error_0, error_1
        end
    end


    return b,cn,ZM_real_space,overlap_vecs,max_overlap,errors
end

function LIOM_real_space(O::Array{Float64,3},bn::Vector{Float64})
    LIOM = O[1,:,:] #O_0

    for n in ProgressBar(2:2:length(bn))
        phi_n=1
        for i in 1:Int(n/2)
            phi_n *=(bn[2*i-1]/bn[2*i])
        end
        LIOM += phi_n*O[n+1,:,:] #O_2n but i+1 because of 0 shift
    end
    return LIOM
end

function cn_calc(vecs,On)
    cns =ComplexF64[]
    for vec in vecs
        P_vec = vec*transpose(vec)
        push!(cns,Op_Inner(On,P_vec))
    end
    return cns
end

function vec_to_krylov(vec::Union{Array{Float64,1},Vector},On)
    
    P_vec = vec*transpose(vec)
    cn=Float64[]
    for n in ProgressBar(1:size(On)[1])
        push!(cn,tr(P_vec*On[n,:,:]) / length(vec))
    end
    return cn
end


function LIOM_from_b(bn)
    #Taking care of different storage conventions, should extend to other functions
    if bn[1]==0
        bn[1]=1
    elseif bn[1]==1
    else 
        bn = vcat(1,bn)
    end

    phi_n = zeros(length(bn))
    phi_n[1]=1 #phi_0
    for n in 3:2:length(bn)
        phi=1
        for i in 1:Int((n-1)/2)
            phi *=(bn[2*i]/bn[2*i+1]) #Because the bn array starts at b2, this ratio is flipped from analytical expression
        end
        phi_n[n]=phi #n+1 because of 0 start
    end
    return phi_n
end

function LIOM_log_avg_from_b(bn_tot)
    Nsamples = size(bn_tot)[2]
    Nsteps = size(bn_tot)[1]+1
    LIOMs=zeros(Nsteps,Nsamples)

    for j in ProgressBar(1:Nsamples)
        LIOMs[:,j]=LIOM_from_b(bn_tot[:,j])
    end
    LIOM_log_avgs=(mean(log.(abs.(LIOMs)),dims=2))
    return LIOM_log_avgs
end

function LIOM_from_zero_mode(bn)
    L = L_matrix(bn)
    #M=(transpose(L)*L)
    vals,vecs=eigen(L)
    vals_abs = abs.(vals)
    min_i = argmin(vals_abs)
    return eigvecs(L)[:,min_i]
end

function LIOM_avg_IPR(bn_tot)
    Nsamples = size(bn_tot)[2]
    IPRs=zeros(Nsamples)
    Norms=zeros(Nsamples)
    for j in ProgressBar(1:Nsamples)
        LIOM = LIOM_from_b(bn_tot[:,j])
        IPRs[j]=sum(abs.(LIOM).^4)/sum(abs.(LIOM).^2)^2
        Norms[j]=sum(abs.(LIOM).^2)
    end
    IPR_avg=exp.(mean(log.(IPRs)))
    #IPR_avg=mean(IPRs)
    Norm_avg = exp(mean(log.(Norms)))
    return IPR_avg, Norm_avg
end
function LIOM_avg_TunnelRate(bn_tot)
    Nsamples = size(bn_tot)[2]
    Gs=zeros(Nsamples)
    for j in 1:Nsamples
        b_n_out =bn_tot[end,j]
        LIOM = LIOM_from_b(bn_tot[1:end-1,j])
        LIOM /= sqrt(sum(abs.(LIOM).^2)) #normalize
        Gs[j]=(b_n_out*LIOM[end])^2
        if Gs[j]==0.
            println("LIOM[end] is zero for sample $j")
            #deleteat!(Gs, j)
        end
    end
    G_avg=exp.(mean(log.(Gs)))
    return G_avg
end
function LIOM_avg_final_val(bn_tot)
    Nsamples = size(bn_tot)[2]
    Gs=zeros(Nsamples)
    for j in 1:Nsamples
        LIOM = LIOM_from_b(bn_tot[:,j])
        LIOM /= sqrt(sum(abs.(LIOM).^2)) #normalize
        Gs[j]=maximum((LIOM[end],LIOM[end-1]))
    end
    G_avg=exp.(mean(log.(Gs)))
    #G_avg = mean(Gs)
    return G_avg
end


function LIOM_from_xs(bn_tot)
    K=size(bn_tot)[2]
    xs=x_generate(bn_tot,K)
    xs_cumsum = cumsum(xs,dims=1)
    xs_mean_cumsum=mean(xs_cumsum,dims=2)[:]
    LIOM=exp.(-1 .*(xs_mean_cumsum))
    return LIOM
end

function LIOM_from_xs_cumsum(xs_mean_cumsum)
    LIOM=exp.(-1 .*(xs_mean_cumsum))
    return LIOM
end

function phi_eta_calc(bn_tot,K)
    arr_length = length((bn_tot[:,1])[1:2:end])-1
    xs = zeros(arr_length,K)
    ys = zeros(arr_length,K)
    if K>size(bn_tot)[2]
        println("K larger than bn_tot[2]")
    else
        for k in 1:1:K
            @show k
            bn = bn_tot[1:end,k]
            x_is = Float64[]
            y_is = Float64[]
            for i in 1:1:arr_length
                x_i = -log(bn[2i-1]/bn[2i])
                y_i = -log(bn[2i]/bn[2i+1])
                push!(x_is,x_i)
                push!(y_is,y_i)
            end
                xs[:,k] = x_is
                ys[:,k]= y_is
        end
    end
    xs_cumsum = cumsum(xs,dims=1)
    ys_cumsum = cumsum(ys,dims=1)
    xs_mean_cumsum=mean(xs_cumsum,dims=2)[:]
    ys_mean_cumsum=mean(ys_cumsum,dims=2)[:]

    phi=exp.(-1 .*(xs_mean_cumsum))
    eta = exp.(-1 .*(ys_mean_cumsum))
    return phi,eta
end

function M_mom_calc(bs,moments,two_k,m)

    #bs should start with index 1
    #Indexing starts at -1
    #moments should start with index 1 and include odd terms, even if set to zero

    if m==-1
        M_m_2k = 0
    elseif m==0
        M_m_2k = moments[two_k]
    else
        M_m_2k = (M_mom_calc(bs,moments,two_k,m-1)/bs[m-1+2]^2)-(M_mom_calc(bs,moments,two_k-2,m-2)/bs[m-2+2]^2)
    end
    return M_m_2k
end

function moments_from_b(bn)
    if bn[1] == 0. || bn[1] ==1.
        bn = bn[2:end]
    end
    L=Tridiagonal(bn,zeros(length(bn)+1), bn)

    return [(L^n)[1,1] for n in 1:2*length(bn)]
end

function b_from_moments(m_vec,K=length(m_vec[2:2:end]))
    bs = Float64[1,1]
    for n in 1:K
        M = M_mom_calc(bs,m_vec,2n,n)
        if M<0
            error("M=$M <0 at n=$n")
        end
        push!(bs,sqrt(M))
    end

    return bs[3:end]
end
