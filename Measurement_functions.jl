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
function gap_find(e_vals)
    sorted_vals = sort!(e_vals)
    idx = argmin(abs.(sorted_vals))
    lower = idx > 1 ? sorted_vals[idx - 1] : nothing
    center = sorted_vals[idx]
    upper = idx < length(sorted_vals) ? sorted_vals[idx + 1] : nothing

    return (lower, center, upper)
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


function FCS_moment(n::Int,N::Int,Q_t::Union{Vector{Matrix{ComplexF64}},Vector{SparseMatrixCSC{ComplexF64, Int64}}},Qs::Union{Vector{Matrix{ComplexF64}},Vector{SparseMatrixCSC{ComplexF64, Int64}}})
    #Need to input one vector of {UTQ^jU_T}_n>j>1 and another of {Q^j}
    if length(Q_t)<n/2
        error("Incorrect number of FCS powers (need n/2)")
    end

    if iseven(n)==false
        return 0
    else
        moment=2*Qs[n] #j=0 term
        #@show 1,tr(moment)
        moment += ((-1)^Int(n/2)) * n_choose_k(n,Int(n/2)) * (Q_t[Int(n/2)]*Qs[Int(n/2)])
        #@show 2,tr(moment)
        for j in 1:Int(n/2)-1 
            moment += 2 * ((-1)^j) * n_choose_k(n,j) * (Q_t[j]*Qs[n-j])
         #   @show 3,tr(moment)
        end

        return tr(moment)/2^N
    end
end

function moment(n::Int,N::Int,Q_t::Union{Vector{Matrix{ComplexF64}},Vector{SparseMatrixCSC{ComplexF64, Int64}}},Qs::Union{Vector{Matrix{ComplexF64}},Vector{SparseMatrixCSC{ComplexF64, Int64}}})
    if iseven(n) == false
        return 0
    else
        M =  Qs[n] + Q_t[n] #nth & 0th term
        for j in 1:n-1
            M += ((-1)^j) * n_choose_k(n,j) * Q_t[n-j] * Qs[j]
        end
        return tr(M)/2^N
    end
end

function moment_eq19(n::Int,N::Int,Ut,U_t,Q)
    if iseven(n) == false
        return 0
    else
        M =  (-1)^(n/2) * n_choose_k(n,Int(n/2)) *Ut*Q^(n/2) *U_t * Q^(n/2) #nth 
        M += 2*Q^n #0th term
        for j in 1:n-1
            M += ((-1)^j) * n_choose_k(n,j) * Ut*Q^(j)*U_t * Q^(n-j)
        end
        return tr(M)/2^N
    end
end

 comm(A,B) = A*B - B*A

function moment_comm(n::Int,N::Int,Ut,U_t,Q::Union{Matrix{ComplexF64},SparseMatrixCSC{ComplexF64, Int64}})
    if iseven(n) == false
        return 0
    else
        M_ = comm(Q,U_t)
        for i in 2:n
            M_ = comm(Q,M_)
        end
    end
    M = Ut*M_
    return tr(M)/2^N
end

function Correlator_ED_t(t_vec::Union{Vector,StepRangeLen,StepRange{Int64, Int64}},vals::Vector,vecs::Matrix,O::AbstractMatrix)
    C_t_vec = Vector{Float64}(undef, length(t_vec))

    Op_eigen = vecs' * O * vecs
    Op_eigen_sq = abs2.(Op_eigen)
    e_diff = (vals .- vals')
    
    for (i,t) in enumerate(t_vec)
        phase_t = exp.(im .* e_diff .*t)
        Prod = (phase_t * Op_eigen_sq)
        C_t_vec[i]=real(tr(Prod))
    end

    return C_t_vec
end

function Spectral_fct_ED(H::AbstractMatrix,Os::Vector{AbstractMatrix},e_start::Int64=1,e_end::Int64=size(H)[1])
    
    vals,vecs = eigen(H)

    Os_eig = [vecs' * O * vecs for O in Os]

    freqs = Float64[]
    peaks = [Float64[] for _ in 1:length(Os)]

    vecs = vecs[:,e_start:e_end]
    vals = vals[e_start:e_end]

    for n in eachindex(vals)
        for m in eachindex(vals)
            push!(freqs,real(vals[n]-vals[m]))
            for (i, Oeig) in enumerate(Os_eig)
                push!(peaks[i], abs(Oeig[n, m])^2)
            end        
        end
    end
    return freqs, peaks
end
function Spectral_fct_ED(H::AbstractMatrix,O::AbstractMatrix,e_start::Int64=1,e_end::Int64=size(H)[1])
    
    vals,vecs = eigen(H)

    O_eig = vecs' * O * vecs

    freqs=Float64[]
    peaks = Float64[]; 

    vecs = vecs[:,e_start:e_end]
    vals = vals[e_start:e_end]


    for n in eachindex(vals)
        for m in eachindex(vals)

            push!(freqs,real(vals[n]-vals[m]))
            push!(peaks,abs(O_eig[n,m])^2)
        end
    end
    return freqs, peaks
end

function Spectral_fct_ED(H::AbstractMatrix,O::Array{<:Any,3})
    
    vals,vecs = eigen(H)

    O_eig = zeros(size(O)) #not defined for complex H right now
    for i in eachindex(size(O)[3])
        O_eig[:,:,i] = vecs' * O[:,:,i] * vecs
    end
    O_eig = mean(O_eig,dims=3)[:,:]

    #normalize so that peaks are of same order as bare Z operators for convenience
    O_eig /= norm(O_eig)

    freqs = Float64[]
    peaks = Float64[]

    for n in eachindex(vals)
        for m in eachindex(vals)
            push!(freqs,real(vals[n]-vals[m]))
            push!(peaks,abs(O_eig[n,m])^2)
        end
    end
    return freqs, peaks
end




function bin_indices(x::Vector{<:Real}, edges::Vector{<:Real})
    nbins = length(edges) - 1
    bins = Int.(zeros(length(x)))

    for i in eachindex(x)
        xi = x[i]

        for b in 1:nbins

            if edges[b] ≤ xi < edges[b+1] #find which bin xi belongs to 
                bins[i] = b #label the bin for the point 
                break
            end
        end
        if bins[i]==0
            error("error, value ($xi) outside of edges ($(edges[1]) to $(edges[end]))")
        end
    end

    return bins
end

function Bin_heights(x::Vector{<:Real},y::Vector{<:Real},edges::Vector{<:Real})


    bin_inds = bin_indices(x, edges)
    # ACTION there are zeros in the bin_idx, fix that!
    binned_y = zeros(length(edges) - 1)
    for (i, bin_idx) in enumerate(bin_inds)
        binned_y[bin_idx] += y[i]
    end

    return binned_y
end
function Bin_heights(x::Vector{<:Real},y::Vector{<:AbstractVector{<:Real}},edges::Vector{<:Real})


    bin_inds = bin_indices(x, edges)
    binned_y = [zeros(length(edges) - 1) for _ in 1:length(y)]  # One array per operator

    for (i, bin_idx) in enumerate(bin_inds)
        for (j, yj) in enumerate(y)
            binned_y[j][bin_idx] += yj[i]
        end
    end

    return binned_y
end


#=
function moment_wf_calc(n::Int,Qs_IT::Vector{MPO},psi1_0::MPS,psi1_t::MPS,psi2_0::MPS,psi2_t::MPS)
    if iseven(n) == false
        return 0
    else
        M =  (inner(psi2_0,Qs_IT[n],psi1_0) + inner(psi1_t, Qs_IT[n],psi2_t))* inner(psi1_0,psi2_0) #nth term
        for j in 1:n-1
            M += ((-1)^j) * n_choose_k(n,j) * inner(psi1_t, Qs_IT[n-j], psi2_t) * inner(psi2_0, Qs_IT[j], psi1_0)
        end
        return M
    end
end
=#
