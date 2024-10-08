include("Pauli Generator.jl")
include("H_building.jl")

function Z_measure(c_vec,Basis_M)
    psi=abs2.(c_vec)*Basis_M
return psi
end

function sparsity(M)
    return length(M.nzval)/length(M)
end

function filter_dist(e_n,e_avg,Gamma)
    eta=0.5 #Effective fraction of eigenstates included in SFF, set to 0.5
    return  exp((e_n-e_avg)^2 / (2*eta*Gamma)^2) #Gamma^2 is variance
end

function SFF_iter(tau,H_Sparse)

    H=Matrix(H_Sparse)
    eig_vals = eigvals(Matrix(H))
    e_avg = mean(eig_vals)
    Gamma = var(eig_vals)

    SFF_sum=0

    for e_n in eig_vals

        SFF_sum += exp(-im*2*pi*e_n*tau)*filter_dist(e_n,e_avg,Gamma)
            
    end

    return abs(SFF_sum)^2
end

function SFF_QS_avg(tau,K)

    SFF_t = Float64[]

    for k in 1:K
        H_Sparse=QuantumSunH(N,L,alpha,xi_i,spinflag)

        SFF_t = SFF_iter(tau,H_Sparse)
    end

    return mean(SFF_t)
end


function Heisenberg_time_QS(N,L,alpha,xi_i,spinflag,K)

    Gamma_0_sq_list = Float64[]
    D = 2^(L+N)

    for k in ProgressBar(1:K)

        QS_H=Matrix(QuantumSunH(N,L,alpha,xi_i,spinflag))
        Gamma_0_sq_itr = (tr(QS_H^2)/D - tr(QS_H)^2 /D^2)
        push!(Gamma_0_sq_list,Gamma_0_sq_itr)
    end

    avg_Gamma_0_sq = mean(Gamma_0_sq_list)

    chi=0.3413 #assuming gaussian density of states, gives percent of states we consider
    dE_avg = sqrt(avg_Gamma_0_sq)/(chi*D) 
    t_H = 1/dE_avg

    return t_H
end


function Level_spacings(H,spectrum_flag)
    #using 
    H_full = Matrix(H)
    vals = eigvals(H_full)
    spacings = Float64[]
    for i in 1:length(vals)-1
        push!(spacings,vals[i+1]-vals[i])
    end

    dimH = length(vals)
    Half_dimH = Int(round(dimH/2))
    mid_N = Int(round(minimum([dimH/10,500])/2))

    if spectrum_flag == "all"
        return spacings
    elseif spectrum_flag == "mid"
        return spacings[Half_dimH-mid_N:Half_dimH+mid_N] 
    end
end

function Level_ratios(H,spectrum_flag)
    #using 
    spacings = Level_spacings(H,spectrum_flag)
    ratios = Float64[]
    for i in 1:length(spacings)-1
        r = spacings[i+1]/spacings[i]
        if r>1
            push!(ratios,1/r)
        else
            push!(ratios,r)
        end
    end
    return ratios
end

auto_corr(A,B,N) = tr(A*B)/2^N
n_choose_k(n::Int,k::Int) = factorial(n)/(factorial(k) * factorial(n-k))

function FCS_moment(n::Int,N::Int,UQU::Vector{MPO},Qs::Vector{MPO})
    #Need to input one vector of {UTQ^jU_T}_n>j>1 and another of {Q^j}
    if length(UQU)<n/2
        error("Incorrect number of FCS powers (need n/2)")
    end

    if iseven(n)==false
        return 0
    else
        moment=2*Qs[n] #j=0 term
        @show tr(moment)
        moment += ((-1)^Int(n/2)) * n_choose_k(n,Int(n/2)) .* apply(UQU[Int(n/2)],Qs[Int(n/2)])
        @show tr(moment)
        for j in 1:Int(n/2)-1 
            moment += 2 * ((-1)^j) * n_choose_k(n,j)*apply(UQU[j],Qs[n-j])
            @show tr(moment)
        end

        return tr(moment)/2^N
    end
end
