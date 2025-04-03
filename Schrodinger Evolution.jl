#A code to generate basic Schrodinger time evolution
using ProgressBars
include("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//General_Quantum//Pauli Generator.jl")

function diag_TE_measure(psi::Union{Vector{ComplexF64},Vector{Float64}},psi_dag::Union{Vector{ComplexF64},Vector{Float64}},H::Union{Matrix{ComplexF64},Matrix{Float64}},t_array::Union{Vector{Float64},Vector{Int64}},Ops)
    eigH=eigen(H)
    Measure_t_array=Matrix{ComplexF64}(undef, length(t_array), length(Ops))

    c_n = eigH.vectors' * psi  # Project psi onto eigenbasis
    c_n_dag = eigH.vectors' * psi_dag  # Project psi onto eigenbasis

    for (ti,t) in ProgressBar(enumerate(t_array))
        phase_factors = exp.(-im * t .* eigH.values)
        psi_t = eigH.vectors * (phase_factors .* c_n)
        psi_dag_t = eigH.vectors * (phase_factors .* c_n_dag)

        Measure_t_array[ti,:] = ([(psi_dag_t'*Ops[i]*psi_t) for i in eachindex(Ops)])

    end
    return Measure_t_array
end

function diag_TE(psi, H,t_array)
    eigH = eigen(H)
    c_n = eigH.vectors' * psi  # Project psi onto eigenbasis
    psi_t_array = Matrix{ComplexF64}(undef, length(t_array), length(psi))

    for (ti, t) in ProgressBar(enumerate(t_array))
        phase_factors = exp.(-im * t .* eigH.values)
        psi_t = eigH.vectors * (phase_factors .* c_n)
        psi_t_array[ti, :] = psi_t
    end

    return psi_t_array
end

function exp_TE_measure(psi::Union{Vector{ComplexF64},Vector{Float64}},psi_dag::Union{Vector{ComplexF64},Vector{Float64}},H::Union{Matrix{ComplexF64},Matrix{Float64}},t_array::Union{Vector{Float64},Vector{Int64}},Ops)
    N = Int(log2(size(H)[1]))
    Measure_t_array=Matrix{ComplexF64}(undef, length(t_array), length(Ops))
    for (ti,t) in ProgressBar(enumerate(t_array))

        expH = exp(-im*t*Matrix(H))
        psi_t = expH*psi
        psi_dag_t = expH*psi_dag

        Measure_t_array[ti,:] = ([(psi_dag_t'*Ops[i]*psi_t) for i in eachindex(Ops)])

    end
    return Measure_t_array
end

function diag_ZZcorr(psi::Union{Vector{ComplexF64},Vector{Float64}},H::Union{Matrix{ComplexF64},Matrix{Float64}},t_array::Union{Vector{Float64},Vector{Int64}})
    eigH=eigen(H)
    N=Int(log2(size(H)[1]))
    S_t = Matrix{ComplexF64}(undef, length(t_array), N)
    Z_vec = Vector{Matrix{ComplexF64}}(undef,N)
    
    V = eigH.vectors
    Vdag = V'  # conjugate transpose

    for n in 1:N
        str = zeros(N);str[n]=3
        Z_vec[n]=pauli_expand(str,"spin")
    end
    
    c_n = Vdag * psi
    c_n_Z = Vdag * Z_vec[1] * psi

    for (ti, t) in enumerate(t_array)
        @show t
        phases = exp.(-im * t .* eigH.values)
        
        psi_t = V*(phases .* c_n)
        psi_Z_t = V*(phases .* c_n_Z)

        S_t[ti,:]=([(psi_t'*Z_vec[i]*psi_Z_t) for i in 1:N])
    end
    return S_t
end

function diag_TE(O::Union{Matrix{ComplexF64},Matrix{Float64}},H_vecs::Union{Matrix{ComplexF64},Matrix{Float64}},H_vals::Union{Vector{Float64},Vector{Int64}},t_array::Union{Vector{Float64},Vector{Int64}})
    O_t_array = []
    for t in t_array 
        O_t=complex(zeros(length(H_vals),length(H_vals)))
        for n in eachindex(H_vals) 
            for m in eachindex(H_vals)
                @show n,m #This is very inefficient
                e_n=H_vals[n] ; vec_n = H_vecs[:,n]
                e_m=H_vals[m] ; vec_m = H_vecs[:,m]
                O_t += exp(im*t*(e_n-e_m))*(vec_n' * O * vec_m)* (vec_n * vec_m')
        
            end
        end
        push!(O_t_array,O_t)
    end
    return O_t_array
end

