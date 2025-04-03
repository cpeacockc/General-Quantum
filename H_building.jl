using Random, Statistics, LinearAlgebra,SparseArrays

include("Pauli Generator.jl")

function GOE_H(N)
    R_ = randn(2^N,2^N)
    R = (1/sqrt(2)) .* (R_ + transpose(R_)) #Random GOE matrix
    return R
end

function Random_XYZ(N)
    Field_vec=ones(3)
    Int_vec=ones(3)
    spinflag="pauli"

    Big_Op=spzeros(2^N,2^N)
        
    for i in 1:N
        for j in 1:N
            if i<j
                str = zeros(N)
                str[i]=1;str[j]=1 #X_i X_i+1
                Big_Op += Int_vec[1]*randn() .* pauli_expand(str,spinflag)


                str = zeros(N)
                str[i]=2;str[j]=2 #Y_i Y_i+1
                Big_Op += Int_vec[2]*randn() .* pauli_expand(str,spinflag)

                str = zeros(N)
                str[i]=3;str[j]=3 #Z_i Z_i+1
                Big_Op += Int_vec[3]*randn() .* pauli_expand(str,spinflag)
            end
        end

        str = zeros(N)
        str[i]=1;#X_i
        Big_Op += Field_vec[1]*randn() .* pauli_expand(str,spinflag)

        str = zeros(N)
        str[i]=3;#Z_i
        Big_Op += Field_vec[3]*randn() .* pauli_expand(str,spinflag)

    end
        
    return real(Big_Op).*( 2^N / ((N/2)*(3N+1))) #This normalization factor makes bandwidth scaling equal to GOE

end




function UltraMetric_H(N,L,alpha,J)

    #First we create the initial N site GOE and project it into full Hilbert space
    gamma=1 #as used in similarity btwn QS and UM model...
    R = GOE_H(N) #Returns (1/sqrt(2)) .* (R_ + transpose(R_)) with R drawn from normal dist var=1 mean=0

    Big_Op = (gamma .* R) ./ (sqrt((2^N) + 1)) # normalize initial GOE by sqrt(2^N +1)
    Id = diagm([1,1])

    for i in 1:L
        Big_Op = kron(Id,Big_Op) #project initial 2^N size GOE into 2^(L+N) Hilbert space.
    end

    GOE = copy(Big_Op)

    #Next we create the sum of block diagonal parts 

    for k in 1:L
        Hk = zeros(2^(N+L),2^(N+L)) #Initialize Hk 

        for i in 1:2^(L-k) # for each R_i

            Id = zeros(2^(L-k),2^(L-k))
            Id[i,i]=1 #Projector such that when kroneckered with in next line, will give block diagonals

            Hk += kron(Id,GOE_H(N+k) ./ (sqrt((2^(N+k)) + 1)))

        end

        Big_Op += J*(alpha^k) .* Hk #normalize Hk by alpha^k and add to full H
    end

    return Big_Op, GOE
end



#Quantum Sun Model
function QuantumSunH(N,L,alpha,spinflag)
    #N is size of random GOE matrix R
    #L is size of spin-1/2 particles connected to R
    #alpha is coupling parameter
    #xi_vec gives exponential localization of spins

    #R_ = randn(2^N,2^N)
    #R = (0.3/2) .* (R_ + transpose(R_)) #Random GOE matrix

    #Choose random fields iid between 0.5-1.5
    h_i = (1.5-0.5) .* rand(L) .+ 0.5
    xi_i=0.2

    gamma=1 #as used in similarity btwn QS and UM model...

    Big_Op = sparse((gamma/sqrt(2^N + 1) .* GOE_H(N)))
    Id = spdiagm([1,1])

    for i in 1:L
        Big_Op = kron(Id,Big_Op)
    end


    #A=0
    for i in 1:L

        #coupling to dot

        #pick random site in dot to connect to
        n_i = rand(1:N)

        str = zeros(N+L)
        str[n_i] = 1 #X on site n_i in R
        str[i+N] = 1 #X on site i in L
        X_iX_n_i = pauli_expand(str,spinflag)

        #xi_i = xi_vec[i]
        u_i = ((i+xi_i)-(i-xi_i))*rand() + (i-xi_i)

        #Z-field term
        str = zeros(N+L) ######## IS IT N+L OR JUST L ################
        str[i+N] = 3 #create Sz op on site i
        Z_i = pauli_expand(str,spinflag)

       # A+=(alpha^u_i) 

        Big_Op += real((alpha^u_i) .* X_iX_n_i + h_i[i] .* Z_i)

    end


    return Big_Op
end


function QuantumSunH_randXYZ(N,L,alpha,spinflag)
    #N is size of random GOE matrix R
    #L is size of spin-1/2 particles connected to R
    #alpha is coupling parameter
    #xi_vec gives exponential localization of spins

    #R_ = randn(2^N,2^N)
    #R = (0.3/2) .* (R_ + transpose(R_)) #Random GOE matrix

    #Choose random fields iid between 0.5-1.5
    h_i = (1.5-0.5) .* rand(L) .+ 0.5
    xi_i=0.2

    #gamma=sqrt(2) #To increase bandwidth of sun since it is random XYZ and not GOE
    gamma=1
    Big_Op = (gamma/sqrt(2^N + 1) .* (Random_XYZ(N)))
    Id = spdiagm([1,1])

    for i in 1:L
        Big_Op = kron(Id,Big_Op)
    end

    #A=0
    for i in 1:L

        #coupling to dot

        #pick random site in dot to connect to
        n_i = rand(1:N)

        str = zeros(N+L)
        str[n_i] = 1 #X on site n_i in R
        str[i+N] = 1 #X on site i in L
        X_iX_n_i = real(pauli_expand(str,spinflag))

        #xi_i = xi_vec[i]
        u_i = ((i+xi_i)-(i-xi_i))*rand() + (i-xi_i)

        #Z-field term
        str = zeros(N+L) 
        str[i+N] = 3 #create Sz op on site i
        Z_i = real(pauli_expand(str,spinflag))

        #A+=(alpha^u_i) 

        Big_Op += real((alpha^u_i) .* X_iX_n_i + h_i[i] .* Z_i)

    end

    #Big_Op *= sqrt((2/sqrt(3))) #normalizing to match quantumsunH - only works for N=3
    return Big_Op
end

    
function H_XYZ(N,J_vec,X_vec,Y_vec,Z_vec,flag)
    Jx=J_vec[1]
    Jy=J_vec[2]
    Jz=J_vec[3]
    Big_Op=spzeros(2^N,2^N)
    
    for i in 1:N-1
        str = zeros(N)
        str[i]=1;str[i+1]=1 #X_i X_i+1
        Big_Op += Jx .* pauli_expand(str,flag)


        str = zeros(N)
        str[i]=2;str[i+1]=2 #Y_i Y_i+1
        Big_Op += Jy .* pauli_expand(str,flag)

        str = zeros(N)
        str[i]=3;str[i+1]=3 #Z_i Z_i+1
        Big_Op += Jz .* pauli_expand(str,flag)
    end
    
    
    for i in 1:N
        str = zeros(N)
        str[i]=1;#X_i
        Big_Op += X_vec[i] .* pauli_expand(str,flag)
    end

    for i in 1:N
    
        str = zeros(N)
        str[i]=2;#Y_i
        Big_Op += Y_vec[i] .* pauli_expand(str,flag)
    end
    
    for i in 1:N
        str = zeros(N)
        str[i]=3;#Z_i
        Big_Op += Z_vec[i] .* pauli_expand(str,flag)
    end
    
    return Big_Op
end
function randanti(N)
    dim = 2^N
    H = randn(dim, dim)  # Random normal matrix
    return H - H'  # Enforce antisymmetry
end
function randH_real(N)
    dim = 2^N
    H = randn(dim, dim)  # Random normal matrix
    return H + H'  # Enforce symmetry
end

H_MBL_basic(N,J,W,flag) = H_XYZ(N,J*ones(3),zeros(N),zeros(N),W .* (rand(MersenneTwister(),N).*2 .-1),flag)

function H_XYZ_PBC(N,J_vec,X_vec,Y_vec,Z_vec,flag)
    Jx=J_vec[1]
    Jy=J_vec[2]
    Jz=J_vec[3]
    Big_Op=spzeros(2^N,2^N)
    
    for i in 0:N-1
        str = zeros(N)
        str[i+1]=1;str[(i +1)%N+1]=1 #X_i X_i+1
        Big_Op += Jx .* pauli_expand(str,flag)


        str = zeros(N)
        str[i+1]=2;str[(i +1)%N+1]=2 #Y_i Y_i+1
        Big_Op += Jy .* pauli_expand(str,flag)

        str = zeros(N)
        str[i+1]=3;str[(i +1)%N+1]=3 #Z_i Z_i+1
        Big_Op += Jz .* pauli_expand(str,flag)
    end
    
    
    for i in 1:N
        str = zeros(N)
        str[i]=1;#X_i
        Big_Op += X_vec[i] .* pauli_expand(str,flag)
    end

    for i in 1:N
    
        str = zeros(N)
        str[i]=2;#Y_i
        Big_Op += Y_vec[i] .* pauli_expand(str,flag)
    end
    
    for i in 1:N
        str = zeros(N)
        str[i]=3;#Z_i
        Big_Op += Z_vec[i] .* pauli_expand(str,flag)
    end
    
    return Big_Op
end

#L must be Even
function H_Build(L_tot,printpath,zeromag)
    cd(printpath)
    tc = Dates.now()
    BinaryArr = Array{Float64}[]
    BinArr_row_i = Int64[]

    @time for i = 0:2^(L_tot)-1
        Bin = UInt(i)
        BinVec = digits(Bin, base=2, pad=L_tot)
            if zeromag=="Zero-mag"
                if sum(BinVec)==L_tot/2                #CHOOSE ZERO MAG SECTOR
                    push!(BinArr_row_i,i)
                    push!(BinaryArr,BinVec)
                end
            else zeromag=="Full"
                push!(BinArr_row_i,i)
                push!(BinaryArr,BinVec)
            end
    end

    #Find Neel state of up, dn, up, dn
    init_vec = zeros(L_tot)
    init_vec[1:2:end] .=1.0
    init_vec_index=findall(x->x==init_vec,BinaryArr)[1]

    M0 = zeros(2^L_tot)


    #Label basis states which are included in zero-mag sector. index i means state associated with binary number i-1
    for i in BinArr_row_i
        M0[i+1]=1
    end

    #Label zero mag basis states in order, but in the full 2^N space
    cmM0 = cumsum(M0)

    #Count how many zero-mag states there are
    M = length(BinArr_row_i)
    I=Int64[]
    J=Int64[]
    @time for i in 1:M #Loop over all basis vectors
        vec = BinaryArr[i] # Consider ith vec
        vec_temp = copy(vec)
        for j in 1:L_tot-1 #Loop over all sites
            if (vec[j]+vec[j+1])==1 #If neighboring spins are diff
            vec_temp[j]=vec[j+1];vec_temp[j+1]=vec[j] #Flip them (S+S- part of H)
            bitN = Int(bitarr_to_int(vec_temp)) #Label which basis bit number this new vector represents
                if bitN in BinArr_row_i #If the new basis vec is still in the zero mag sector
                    append!(I,i); #Create matrix entry connecting the original basis vec...
                    push!(J,cmM0[bitN+1]) #...and the new basis vec (after flipping) (bitN finds bin number in ordered full basis space, which is correctly labeled for new space from cumsum earlier)
                else #Else if its not in zero mag sector don't need an entry
                end
            else #Else if neighoring spins are same, flip does nothing)
            end
            vec_temp = copy(vec) #Reset vector that got flipped
        end
    end


    td = Dates.now()

    fid = h5open("H-L$(L_tot).h5", "w")
        create_group(fid, "H")
        g = fid["H"]
        g["I"] = I
        g["J"] = J
        g["V"] = 0.5*ones(length(I))
        g["bitH_runtime_sec"] = ((td-tc).value)*0.001
        g["BinArr_row_i"]=BinArr_row_i
        g["BinaryArr"] = copy(transpose(reshape(reduce(hcat,BinaryArr),L_tot,:)))
        g["init_vec_index"] = init_vec_index
    close(fid)

    V = 0.5*ones(length(I))


    return (sparse(I,J,V))#normalize by W

end


function H_MBL(L,L_rr,W,W_rr,d,seed,K,k,bitH_path)
    cd(bitH_path);

    if L_rr==0
        L_tot=L
        h = (rand(MersenneTwister(seed),K,L_tot).*2 .-1)[k,:]
        h[1:L]*=W; 
    else
        L_tot=L+L_rr
        h = (rand(MersenneTwister(seed),K,L_tot).*2 .-1)[k,:]
        h[1:L]*=W; h[(L+1):L_tot]*=W_rr
    end

    fid = h5open("H-L$(L_tot).h5","r")
        g = fid["H"]
        I = read(g,"I")
        J = read(g,"J")
        V = read(g,"V")
        init_vec_index = read(g,"init_vec_index")
        bitH_runtime_sec = read(g,"bitH_runtime_sec")
        M = length(read(g,"BinArr_row_i"))
        diagH = Float64[]

    for j in 1:M
        row_Z = (read(g,"BinaryArr")[j,:].-0.5)
        HZ = dot(row_Z,h)
        HZZ = d*dot(row_Z[1: end-1],row_Z[2:end])
        push!(diagH,HZ + HZZ)
    end


    H_Sparse = (sparse(I,J,V,M,M)+spdiagm(diagH)) #normalize by W

    c0 = zeros(M);c0[init_vec_index]=1.0
    #c0=complex(c0)

    close(fid)
    return H_Sparse
end

#Define Sz site operator
function Sz_siteOp(L_tot,site)
    cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//UOH//Outs//bitH");
    fid = h5open("H-L$(L_tot).h5","r")
        g = fid["H"]
        Sz_Op = spdiagm(2. .* read(g,"BinaryArr")[:,site] .- 1.)
    close(fid)
    return (0.5)*Sz_Op
end

function H_LIOM(N,Z_vec,Z_field,ZZ_field,alpha,flag)

    Big_Op=spzeros(2^N,2^N)

#=
   for i in 1:N
        for j in 1:N
            for k in 1:N
                str = zeros(N)
                str[i]=3;str[j]=3;str[k]=3 #Z_i Z_j Z_k
                Big_Op += Z_vec[3]*exp(-alpha*maximum([abs(i-j),abs(i-k),abs(k-j)])) .* pauli_expand(str,flag)
            end
        end
    end
    =#
    #exp(-alpha*abs(i-j)) 

    for i in 1:N-1
        str = zeros(N)
        str[i]=3; str[i+1]=3 #Z_i Z_j
        Big_Op += Z_vec[2]*ZZ_field[i,i+1] .* pauli_expand(str,flag)
    end


    for i in 1:N
        str = zeros(N)
        str[i]=3;#Z_i
        Big_Op += Z_vec[1] * Z_field[i] .* pauli_expand(str,flag)
    end

    return Big_Op
end

function H_Anderson1D(L::Int64,W::Union{Int64,Float64},t::Union{Int64,Float64})
    h = (W/2) .* (rand(L).*2 .-1) #random potentials

    H_Anderson = spdiagm(-1 => (-t)*ones(L-1), 1 => (-t)*ones(L-1), 0=>h)

    H_Anderson[1,L]=-t
    H_Anderson[L,1]=-t
    return H_Anderson
end

function H_Anderson1D(L::Int64,t::Union{Int64,Float64},h::Union{Vector{Float64},Vector{Int64}})
    H_Anderson = spdiagm(-1 => (-t)*ones(L-1), 1 => (-t)*ones(L-1), 0=>h)

    H_Anderson[1,L]=-t
    H_Anderson[L,1]=-t
    return H_Anderson
end

function H_Random_Hop1D(L::Int64,W::Union{Int64,Float64},t::Union{Int64,Float64})
    h1 = (W/2) .* (rand(L-1).*2 .-1) #random potentials

    H_Anderson = spdiagm(-1 => (-t)*h1, 1 => (-t)*h1, 0=>zeros(L))

    H_Anderson[1,L]=-(W/2) * (rand()*2 -1)
    H_Anderson[L,1]=H_Anderson[1,L]
    return H_Anderson
end

function H_Random_Hop1D(L::Int64,W::Union{Int64,Float64},t::Union{Int64,Float64},h::Vector{Float64})
    

    H_Anderson = spdiagm(-1 => (-t)*h, 1 => (-t)*h, 0=>zeros(L))

    H_Anderson[1,L]=-(W/2) .* (rand().*2 .-1)
    H_Anderson[L,1]=H_Anderson[1,L]
    return H_Anderson
end

function H_Anderson1D_biased(L::Int64,W::Union{Int64,Float64},t::Union{Int64,Float64},a::Union{Int64,Float64})

    #a=0 is normal anderson, a>0 biases forward hopping and visa versa
    
    h = (W/2) .* (rand(L).*2 .-1) #random potentials

    H_Anderson = spdiagm(-1 => (-t*exp(-a*im))*ones(L-1), 1 => (-t*exp(a*im))*ones(L-1), 0=>h)

    H_Anderson[1,L]=-t
    H_Anderson[L,1]=-t
    return H_Anderson
end


function H_Anderson1D_h(L::Int64,t::Union{Int64,Float64},h::Union{Vector{Float64},Vector{Int64}})

    H_Anderson = spdiagm(-1 => (-t)*ones(L-1), 1 => (-t)*ones(L-1), 0=>h)

    H_Anderson[1,L]=-t
    H_Anderson[L,1]=-t
    return H_Anderson
end

function H_Anderson3D(L,W,t)

    H = spzeros(L^3,L^3)
    Id = spdiagm(ones(L))
        Big_Op = H_Anderson1D(L,0,t)
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,H_Anderson1D(L,0,t))
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,H_Anderson1D(L,0,t))
        H += Big_Op

    h = (W/2) .* (rand(L^3).*2 .-1)
    H += spdiagm(h)
    return H
end

function H_Anderson4D(L,W,t)

    H = spzeros(L^4,L^4)
    Id = spdiagm(ones(L))
    
        Big_Op = H_Anderson1D(L,0,t)
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,H_Anderson1D(L,0,t))
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,H_Anderson1D(L,0,t))
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,Id)
        Big_Op = kron(Big_Op,H_Anderson1D(L,0,t))
        H += Big_Op

    h = (W/2) .* (rand(L^4).*2 .-1)
    H += spdiagm(h)
    return H
end

function H_Anderson2D(L,W,t)

    H = spzeros(L^2,L^2)
    Id = spdiagm(ones(L))
        Big_Op = H_Anderson1D(L,0,t)
        Big_Op = kron(Big_Op,Id)
        H += Big_Op

        Big_Op = Id
        Big_Op = kron(Big_Op,H_Anderson1D(L,0,t))
        H += Big_Op
        
    h = (W/2) .* (rand(L^2).*2 .-1)
    H += spdiagm(h)
    return H
end

function H_X_XX(L::Int64)

    Big_Op = spzeros(2^L,2^L)

    for i in 1:L-1
        str=zeros(L)
        str[i]=1 #X_i
        str[i+1]=1 #X_i+1
        Big_Op += pauli_expand(str,"pauli")

        str=zeros(L)
        str[i]=2 #Y_i
        str[i+1]=2 #Y_i+1
        Big_Op += pauli_expand(str,"pauli")
    end

    return Big_Op
end

function X_all_probe(L::Int64)

    Big_Op = spzeros(2^L,2^L)

    for i in 1:L
        str=zeros(L)
        str[i]=1 #X_i
        Big_Op += pauli_expand(str,"pauli")
    end

    return Big_Op
end

function ZZ_all_probe(L::Int64)

    Big_Op = spzeros(2^L,2^L)

    for i in 1:L-1
        str=zeros(L)
        str[i]=3 #Z_i
        str[i+1]=3 #Z_i
        Big_Op += pauli_expand(str,"pauli")
    end

    return Big_Op
end

function XXZnnn_H(L::Int64)

    H = spzeros(2^L,2^L)
    Δ = 2
    γ = 1/2
    for i in 0:L-1

        str=zeros(L); str[i+1]=1 ;str[(i+1)%L+1]=1 #X_i*X_i+1
        H += pauli_expand(str,"pauli")

        str=zeros(L); str[i+1]=1 ;str[(i+2)%L+1]=1 #X_i*X_i+2
        H += γ .* pauli_expand(str,"pauli")

        str=zeros(L); str[i+1]=2 ;str[(i+1)%L+1]=2 #Y_i*Y_i+1
        H += pauli_expand(str,"pauli")

        str=zeros(L); str[i+1]=2 ;str[(i+2)%L+1]=2 #Y_i*Y_i+2
        H += γ .* pauli_expand(str,"pauli")

        str=zeros(L); str[i+1]=3 ;str[(i+1)%L+1]=3 #Z_i*Z_i+1
        H += Δ .* pauli_expand(str,"pauli")

        str=zeros(L); str[i+1]=3 ;str[(i+2)%L+1]=3 #Z_i*Z_i+2
        H += (Δ *γ) .* pauli_expand(str,"pauli")
    end
    return H
end

function H_Quantum_Ising(L::Int64,hx)

    Big_Op = spzeros(2^L,2^L)

    for i in 1:L-1
        str=zeros(L)
        str[i]=1 #X_i
        str[i+1]=1 #X_i+1
        Big_Op += pauli_expand(str,"pauli")
    end

    #add periodic BCs
    str=zeros(L)
    str[1]=1
    str[L]=1
    Big_Op += pauli_expand(str,"pauli")

    for i in 1:L
        str=zeros(L)
        str[i] = 1 #X_i
        Big_Op += hx * pauli_expand(str,"pauli")

        str=zeros(L)
        str[i] = 3 #Z_i
        Big_Op += -1.05 * pauli_expand(str,"pauli")
    end

    return Big_Op
end