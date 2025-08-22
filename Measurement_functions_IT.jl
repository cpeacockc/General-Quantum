include("Pauli Generator.jl")
using ITensorMPS

function FCS_moment(n::Int,N::Int,Q_t_IT::Vector{MPO},Qs_IT::Vector{MPO})
    #Need to input one vector of {UTQ^jU_T}_n>j>1 and another of {Q^j}
    if length(Q_t_IT)<n/2
        error("Incorrect number of FCS powers (need n/2)")
    end

    if iseven(n)==false
        return 0
    else
        moment=2*Qs_IT[n] #j=0 term
        #@show 1,tr(moment)
        moment += ((-1)^Int(n/2)) * n_choose_k(n,Int(n/2)) * apply(Q_t_IT[Int(n/2)],Qs_IT[Int(n/2)])
        #@show 2,tr(moment)
        for j in 1:Int(n/2)-1 
            moment += 2 * ((-1)^j) * n_choose_k(n,j) * apply(Q_t_IT[j],Qs_IT[n-j])
        #    @show 3,tr(moment)
        end

        return tr(moment)/2^N
    end
end

function moment(n::Int,N::Int,Q_t_IT::Vector{MPO},Qs_IT::Vector{MPO})
    if iseven(n) == false
        return 0
    else
        M =  Qs_IT[n] + Q_t_IT[n] #nth term
        for j in 1:n-1
            M += ((-1)^j) * n_choose_k(n,j) * apply(Q_t_IT[n-j], Qs_IT[j])
        end
        return tr(M)/2^N
    end
end

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

