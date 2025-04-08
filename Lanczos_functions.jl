using HDF5, LinearAlgebra, SparseArrays,Dates,Random,ProgressBars, ITensors, ITensorMPS
include("H_building.jl")
include("Pauli Generator.jl")
include("Measurement_functions.jl")
#include("pauli_strings.jl")


function Lanczos(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64)


    #Base vector
    O = []
    b = Float64[0]
    #Define O0
    Probe/=Op_Norm(Probe)
    push!(O,Probe)
    LO_0 = L_n(O[1],H,2)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    
    #Define O1
    push!(O,LO_0/b[2])
    
    for n in 3:Nsteps
        A_n = L_n(O[2],H,n) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        push!(b,b_n)
        O[1]=O[2]
        O[2] = (A_n/b_n)
        #println("n=$n, bn=$b_n")
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
    return b[2:end],runtimes
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

    println("calculating eigenvectors...")
    e_vecs=eigvecs(H)

    #Find eigenvalue that has the maximum overlap with the initial probe
    println("calculating overlaps...")
    max_overlap_i = []
    max_overlap = Float64[]
    overlap_vecs = []

    for i in 1:1:size(Probe)[1]
        vec = e_vecs[:,i]
        P_vec = vec*transpose(vec)
        overlap=Op_Inner(P_vec,Probe)
        if overlap>0.01 #if the overlap is significant
            @show i, overlap
            push!(max_overlap_i,i)
            push!(max_overlap,overlap)
            push!(overlap_vecs,vec)
        end
    end

    e_vecs=nothing

    cn=zeros(length(overlap_vecs),Nsteps+1)

    #Base vector
    println("starting Lanczos...")
    O = []
    b = Float64[0]

    #Define O0
    push!(O,Probe)
    cn[:,1]=cn_calc(overlap_vecs,O[1])

    LO_0 = L_n(O[1],H,2)
    #Define b1, b0 is set to 0
    push!(b,Op_Norm(LO_0))
    
    #Define O1
    push!(O,LO_0/b[2])
    cn[:,2]=cn_calc(overlap_vecs,O[2])
    
    for n in ProgressBar(3:Nsteps+1)
        A_n = L_n(O[2],H,n) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        push!(b,b_n)
        O[1] = O[2]
        O[2] = A_n/b_n
        cn[:,n]=cn_calc(overlap_vecs,O[2])
    end


    return b[2:end],cn,overlap_vecs,max_overlap
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
    cns=[]
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
    phi_n = zeros(length(bn)+1)
    phi_n[1]=1 #phi_0
    for n in 2:2:length(bn)
        phi=1
        for i in 1:Int(n/2)
            phi *=(bn[2*i-1]/bn[2*i])
        end
        phi_n[n+1]=phi #n+1 because of 0 start
    end
    return phi_n
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