
comm(A,B) = A*B - B*A

using LinearAlgebra

function S_calc(H0,V)
    N = size(H0)[1]
    eigval,eigvec=eigen(H0) #diagonalize H0
    
    #We now want to create a matrix  1/(e_n-e_m)
    em = repeat(eigval,1,N) #create matrix of eigenvalues e_m
    en = em'
    diff_e = en - em
    
    # Fix degenerate spectrum
    diff_e ./= (diff_e .^ 2 ) #create difference matrix

    S = eigvec' * V * eigvec .* diff_e #write V in eigenbasis
    S = triu(S)  # Upper triangular part
    S[diagind(S)] .= 0 

    return eigvec * (S - S') * eigvec'

end

function test_get_S()
    N = 4
    H = randH_real(N)
    S = randanti(N)  

    V = comm(S,H)
    S2 = S_calc(H, V)
    V2 = comm(S2,H)
    println(norm(V - V2))
end

function schrieffer_wolff(H::Union{Matrix{Float64},Matrix{Int64}},iterations::Int64)

    for i in 1:iterations 
        H0 = diagm(LinearAlgebra.diag(H))
        V = H - H0
        S = -S_calc(H0,V)
        #comSH = comm(S,H)
        H = H0 + comm(S,V)/2

        @show norm(V - comm(S,H0))
    end
    return H
end

function schrieffer_wolff(H::Operator64,iterations::Int64)

    for i in 1:iterations 
        H0 = diagm(LinearAlgebra.diag(H))
        V = H - H0
        S = -S_calc(H0,V)
        #comSH = comm(S,H)
        H = H0 + comm(S,V)/2

        @show norm(V - comm(S,H0))
    end
    return H
end


function schrieffer_wolff_alpha2(H,iterations)

    N = size(H)[1]
    alpha=0.01

    for i in 1:iterations 
        H0 = diagm(LinearAlgebra.diag(H))
        V = H - H0
        S = -S_calc(H0,V)
        #comSH = comm(S,H)
        H = (1-alpha)*H + alpha* (H0 + comm(S,H0)/2)

        @show norm(V - comm(S,H0))
    end
    return H
end

function schrieffer_wolff_alpha1(H,iterations)

    N = size(H)[1]
    alpha=0.01

    for i in 1:iterations 
        H0 = diagm(LinearAlgebra.diag(H))
        V = H - H0
        S = -alpha*S_calc(H0,V)
        comSH = comm(S,H)
        H += 0.5 * comSH

        @show norm(V - comm(S,H0))
    end
    return H
end
