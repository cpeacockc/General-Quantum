using Random, Statistics, LinearAlgebra, ITensors

include("Pauli Generator.jl")
include("IT_functions.jl")

function Sz_site_IT(site,s)
    os = OpSum()
    os .+= 1,"Sz",site
    Sz = MPO(os,s)
    return Sz
end

function MBL_H_IT(L,L_rr,d,W,seed,k,K,s)
    if L_rr==0
        L_tot=L
        h = (rand(MersenneTwister(seed),K,L_tot).*2 .-1)[k,:]
        h[1:L]*=W; 
      else
        L_tot=L+L_rr
        h = (rand(MersenneTwister(seed),K,L_tot).*2 .-1)[k,:]
        h[1:L]*=W; h[(L+1):L_tot]*=W_rr
    end

    os = OpSum()
    for i in 1:L_tot-1
        os .+= (d, "Sz",i,"Sz",(i+1))
        os .+= (0.5, "S+",i,"S-",(i+1))
        os .+= (0.5, "S-",i,"S+",(i+1))
    end
    for i in 1:L_tot
        os .+= (h[i],"Sz",i)
    end

    return MPO(os,s)/W
end

MBL_H_IT_basic(L,W,s)=MBL_H_IT(L,0,1,W,rand(1000:100000),1,1,s)

function Random_XYZ_IT(N,s)

    Field_vec=2 .*ones(3) #2 is to convert spin matrices to pauli 
    Int_vec=4 .*ones(3) #4 is to convert spin matrices to pauli 

    os=OpSum()

    for i in 1:N
        for j in 1:N
            if i<j
                os += (Int_vec[1]*randn(),"Sx",i,"Sx",j)
                os += (-Int_vec[2]*randn(),"iSy",i,"iSy",j)
                os += (Int_vec[3]*randn(),"Sz",i,"Sz",j)
            end
        end
    end

    for i in 1:N
        os += (Field_vec[1]*randn(),"Sx",i)

        os += (Field_vec[3]*randn(),"Sz",i)
    end
    return MPO(os,s) * (( 2^N / ((N/2)*(3N+1))))
end


function QuantumSunH_randXYZ_IT(N::Int64,L::Int64,alpha::Float64,s)
    #N is size of random GOE matrix R
    #L is size of spin-1/2 particles connected to R
    #alpha is coupling parameter
    #xi_vec gives exponential localization of spins

    #Choose random fields iid between 0.5-1.5
    h_i = (1.5-0.5) .* rand(L) .+ 0.5
    xi_i=0.2

    gamma=1 #as used in similarity btwn QS and UM model...

    Big_Op = OpSum()

    #A=0

    
    for i in 1:L

        #coupling to dot

        #pick random site in dot to connect to
        n_i = rand(1:N)

        #xi_i = xi_vec[i]
        u_i = ((i+xi_i)-(i-xi_i))*rand() + (i-xi_i)

       # A+=(alpha^u_i) 

        Big_Op .+= ((alpha^u_i),"Sx",i+N,"Sx",n_i)

        Big_Op .+=  (h_i[i],"Sz",i+N)

    end

    H = +(MPO(Big_Op,s), (gamma/sqrt(2^N + 1) * Random_XYZ_IT(N,s)); alg="directsum")
    return H
end

function H_X_XX_IT(L::Int64,s)
    Big_Op = OpSum()

    for i in 1:L-1
        Big_Op += (4,"Sx",i,"Sx",i+1)
        Big_Op += (-4,"iSy",i,"iSy",i+1)
    end
    H=MPO(Big_Op,s)
    return H
end

function X_all_probe_IT(L::Int64,s)
    Big_Op = OpSum()

    for i in 1:L
        Big_Op += (2,"Sx",i)
    end
    Op=MPO(Big_Op,s)
    return Op
end

function H_CC_IT(L,hx,s)
    H = OpSum()

    for i in 1:L
        H += (-2*1.05,"Sz",i)
        H += (2*hx,"Sx",i)
    end

    for i in 1:L-1
        H += (4,"Sx",i,"Sx",i+1)
    end
    return MPO(H,s)
end

function CC_probe_IT(L,s)
    Op = OpSum()

    for i in 1:L
        Op += (2,"Sz",i)
    end

    for i in 1:L-1
        Op += (4*1.05,"Sx",i,"Sx",i+1)
    end

    return MPO(Op,s)
end

function H_XXX_IT(L,s)

    H = OpSum()

    for i in 1:L-1
        H += (4,"Sx",i,"Sx",i+1)
        H += (4,"Sy",i,"Sy",i+1)
        H += (4,"Sz",i,"Sz",i+1)
    end
    
    return MPO(H,s)
end

function XXX_probe_IT(L,s)

    q=1/128

    Op = OpSum()

    for i in 1:L-1

        Op += (4*exp(im*q),"Sx",i,"Sy",i+1)
        Op += (-4*exp(im*q),"Sy",i,"Sx",i+1)
    end

    return MPO(Op,s)
end

function PXP_H(L,omega,s)
    Op = OpSum()
    gs = MPS(s, "Dn")

    for i in 1:L
        Op += projector(gs)
        Op += (2,"Sx",i)

    end
end

function XXZnnn(N::Int, sites)
    H = OpSum()
    Δ = 2
    γ = 1/2
    for j in 0:N-1
        H += 4,"Sx",j+1,"Sx",(j+1)%N+1
        H += 4,"Sy",j+1,"Sy",(j+1)%N+1
        H += 4*Δ,"Sz",j+1,"Sz",(j+1)%N+1
        H += 4*γ,"Sx",j+1,"Sx",(j+2)%N+1
        H += 4*γ,"Sy",j+1,"Sy",(j+2)%N+1
        H += 4*γ*Δ,"Sz",j+1,"Sz",(j+2)%N+1
    end
    H = MPO(H,sites)
    return H
end

function XXZnnn_gates(N::Int,Δ,γ,tau,s)
    gates = ITensor[]
    for j in 0:(N - 1)
            
        s1 = s[j+1] #Initial site
        s2 = s[(j +1)%N+1] #nearest-neighbor
        s3 = s[(j +2)%N+1] #next-nearest-neighbors

            #Id is neccesary to add ITensors. Is this the best way to do it?
        h = 
          4 * op("Sx", s1) * op("Sx", s2) * op("Id",s3) +
          4 * op("Sy", s1) * op("Sy", s2) * op("Id",s3) +
          4 * Δ * op("Sz", s1) * op("Sz", s2) * op("Id",s3) +
    
          4 * γ * op("Sx", s1) * op("Id",s2) * op("Sx", s3) +
          4 * γ * op("Sy", s1) * op("Id",s2) * op("Sy", s3) +
          4 * Δ * γ * op("Sz", s1) * op("Id",s2) * op("Sz", s3)
    
        push!(gates,exp(-im * tau / 2 * h))
    end
    
    append!(gates, reverse(gates))
    return gates
end

function XXZnn_randfields_gates(N::Int,Δ,h,tau,s) #This function is not validated
    gates = ITensor[]
    for j in 1:(N-1)

        s1 = s[j] #Initial site
        s2 = s[j+1] #nearest-neighbor
            
        #s1 = s[j+1] #Initial site
        #s2 = s[(j +1)%N+1] #nearest-neighbor

            #Id is neccesary to add ITensors. Is this the best way to do it?
        hj = 
          4 * op("Sx", s1) * op("Sx", s2) +
          4 * op("Sy", s1) * op("Sy", s2) +
          4 * Δ * op("Sz", s1) * op("Sz", s2) +
          2 * h[j+1] * op("Sz", s1) * op("Id", s2)
    
        push!(gates,exp(-im * tau / 2 * hj))
    end
    
    append!(gates, reverse(gates))
    return gates
end

function XXZnn_gates(N::Int,J::Union{Float64,Int64},Δ::Union{Float64,Int64},tau::Union{Float64,Int64},s) 
    gates = ITensor[]
    for j in 1:(N-1)

        s1 = s[j] #Initial site
        s2 = s[j+1] #nearest-neighbor
            
        #s1 = s[j+1] #Initial site
        #s2 = s[(j +1)%N+1] #nearest-neighbor

        hj = J*( 
            op("Sx", s1) * op("Sx", s2) +
            op("Sy", s1) * op("Sy", s2) +
            Δ * op("Sz", s1) * op("Sz", s2))
    
        push!(gates,exp(-im * tau / 2 * hj))
    end
    
    append!(gates, reverse(gates))
    return gates
end

function CC_gates(N::Int64,hx::Union{Float64,Int64},tau::Union{Float64,Int64},s)
    gates = ITensor[]
    for j in 0:(N - 1)
            
        s1 = s[j+1] #Initial site
        s2 = s[(j +1)%N+1] #nearest-neighbor
            
        #s1 = s[j+1] #Initial site
        #s2 = s[(j +1)%N+1] #nearest-neighbor

        #Id is neccesary to add ITensors. Is this the best way to do it?
        hj = 
          4 * op("Sx", s1) * op("Sx", s2) +
          2 * hx * op("Sx",s1) * op("Id",s2) + 
          -2 *1.05 * op("Sz",s1) * op("Id",s2)
    
    
        push!(gates,exp(-im * tau / 2 * hj))
    end
    #Add fields on last site
     #hj = 2 * hx * op("Sx",s[N]) * op("Id",s[N-1]) + 
    #      -2 *1.05 * op("Sz",s[N]) * op("Id",s[N-1])
    
    #push!(gates,exp(-im * tau / 2 * hj))
    
    append!(gates, reverse(gates))
    return gates
end