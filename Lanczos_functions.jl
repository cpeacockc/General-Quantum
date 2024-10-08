using HDF5, LinearAlgebra, SparseArrays,Dates,Random,ProgressBars, ITensors
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
    
    for n in ProgressBar(3:Nsteps)
        A_n = L_n(O[2],H,n) - b[n-1]*O[1]
        b_n = Op_Norm(A_n)
        push!(b,b_n)
        O[1]=O[2]
        O[2] = (A_n/b_n)
        #println("n=$n, bn=$b_n")
        @show typeof(A_n)
    end
    return b[2:end]
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
                xs[:,k]= x_is
        end
    end
    return xs
end

function Lanczos_full(Probe::Union{Matrix,DenseMatrix,SparseMatrixCSC},H::Union{Matrix,DenseMatrix,SparseMatrixCSC},Nsteps::Int64)

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