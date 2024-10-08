using Random, Statistics, LinearAlgebra

include("pauli_strings.jl")

import .pauli_strings as ps

function randXYZ_ps(N::Int64,L::Int64)
    H = ps.Operator(N+L)
    for i in 1:N
        for j in 1:N
            if i<j
                H += (randn(),'X',i,'X',j)
                H += (randn(),'Z',i,'Z',j)
                H += (randn(),'Y',i,'Y',j)
            end
        end

        H += (randn(),'X',i)
        H += (randn(),'Z',i)

    end
    return H
end

function QS_randXYZ_ps(N::Int64,L::Int64,alpha::Float64)

    #N is size of random GOE matrix R
    #L is size of spin-1/2 particles connected to R
    #alpha is coupling parameter
    #xi_vec gives exponential localization of spins

    #Choose random fields iid between 0.5-1.5
    h_i = (1.5-0.5) .* rand(L) .+ 0.5
    xi_i=0.2
    L_tot= N+L
    gamma=1 #as used in similarity btwn QS and UM model...

    Big_Op =  ps.Operator(L_tot)

    #A=0

    
    for i in 1:L

        #coupling to dot

        #pick random site in dot to connect to
        n_i = rand(1:N)

        #xi_i = xi_vec[i]
        u_i = ((i+xi_i)-(i-xi_i))*rand() + (i-xi_i)

       # A+=(alpha^u_i) 

        Big_Op += (0.25*(alpha^u_i),"X",i+N,"X",n_i)

        Big_Op +=  (0.5*h_i[i],"Z",i+N)

    end

    H = Big_Op + (gamma/sqrt(2^N + 1) * randXYZ_ps(N,L))
    return H
end