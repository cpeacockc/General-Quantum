using TensorOperations,TensorKit,MPSKit,LinearAlgebra


######### DEFINE FUNCTION TO CREATE H #########

function Schwinger_H_MK(N::Int64,J::Union{Float64,Int64},m_lat::Union{Float64,Int64},w::Union{Float64,Int64},theta::Union{Float64,Int64,Irrational})
    S_x = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    S_z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    S_y = TensorMap(ComplexF64[0 -im; im 0], ℂ^2 ← ℂ^2)
    myid = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)

    chain = fill(ℂ^2, N)

    H_zz = Dict()
    H_xx = Dict()
    H_yy = Dict()
    H_z1 = Dict()
    H_z2 = Dict()
    neelproj = Dict()

    for n in 1:1:N
        neelproj[(n,)] =  (myid + ((-1)^n)*S_z)/2
    end

    for n in 2:1:N-1
        for k in 1:1:n-1
            H_zz[(k,n)] =  (J/2)*(N-n)*4*S_z ⊗ S_z
        end
    end

    for n in 1:1:N-1
        H_xx[(n,n+1)] = 0.5*(w- (((-1)^(n-1))*(m_lat/2)*sin(theta)))*4 * S_x ⊗ S_x
        H_yy[(n,n+1)] = 0.5*(w- (((-1)^(n-1))*(m_lat/2)*sin(theta)))*4 * S_y ⊗ S_y
        for l in 1:n+1
        H_z2[(l,)] = (J/2) * (n%2)*2 * S_z
        end
    end

    for n in 1:1:N
        H_z1[(n,)] = (m_lat*cos(theta)/2)*2*((-1)^(n-1)) * S_z
    end



    H_schwinger = FiniteMPOHamiltonian(chain, H_zz)+FiniteMPOHamiltonian(chain, H_z1)+FiniteMPOHamiltonian(chain, H_z2)+FiniteMPOHamiltonian(chain, H_xx)+FiniteMPOHamiltonian(chain, H_yy)


    return H_schwinger
end

function Neel_state_MK(N::Int64)
    chain = fill(ℂ^2, N)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    myid = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)
    neelproj = Dict()
    physical_space = ℂ^2
    max_bond_dim = ℂ^1

    for n in 1:1:N
        neelproj[(n,)] =  (myid + ((-1)^n)*Z)/2
    end

    nproj = FiniteMPOHamiltonian(chain, neelproj )
    ψ₀ = nproj*FiniteMPS(rand, ComplexF64,N, physical_space, max_bond_dim)
    return ψ₀
end