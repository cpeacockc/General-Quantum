
using PyPlot
using PauliStrings
import PauliStrings as ps
using ProgressBars
import LinearAlgebra as la
import PyPlot as plt
using SparseArrays


function buildA(D::Operator, V::Operator) #Need to find a way to truncate this
    N = D.N
    # S = k_local_part(V * D + V + D, 2; atmost=true)
    S = V * D + V + D
    A = spzeros(ComplexF64, length(S), length(V))
    for i in 1:length(S)
        for j in 1:length(D)
            vc = S.v[i] ⊻ D.v[j]
            wc = S.w[i] ⊻ D.w[j]
            c = (-1)^count_ones(S.v[i] & D.w[j]) - (-1)^count_ones(S.w[i] & D.v[j])
            for k in 1:length(V)
                if (vc == V.v[k] && wc == V.w[k])
                    A[i, k] += (-1)^count_ones(vc & V.w[k]) * c * D.coef[j] * (-1)^ycount(V.v[k], V.w[k])
                end
            end
        end
    end
    return A
end


#COMPARE THIS FUNCTION TO EXACT DIAGONALIZATION
#Does s have multiple solutions? implementation of A is correct

function get_S(D, V) #Is there another way to solve for S? This may be giving bad answer. Can we keep S anti-sym?
    A = sparse(transpose(buildA(D, V)))
    # S = k_local_part(V * D + V + D, 2; atmost=true)
    S = V * D + V + D #all possible string coefficients
    x = A \ V.coef
    S.coef = Array(x) #set coefficints. S is anti-sym
    return S
end


# test that get_S is working
function testgetS()
    println("testing get_S")
    N = 4
    D = rand_local2(N)
    S = 1im * rand_local2(N) # anti hermitian matrix
    V = com(S, D)
    S2 = get_S(D, V) # try to recover S from D and V
    println(ps.opnorm(S2))
    println(ps.opnorm(V))
    println("this should be 0: ", ps.opnorm(ps.com(S2, D) - V)) #This works perfectly
end

function sw_diag(H, iter)
    alpha = 0.01
    Hi = deepcopy(H)
    for i in 1:iter
        Hi = trim(Hi, 2^12)
        D = ps.diag(Hi)
        V = Hi - D
        S = -get_S(D, V)
        println(ps.opnorm(com(S, D) - V)) #This is not zero, not working, not finding best S
        println(ps.opnorm(V) / ps.opnorm(Hi)) #This is going to zero, so its working somehow?
        Hi += alpha * 0.5 * com(S, Hi)
    end
    return Hi
end


testgetS()

N = 5
H = rand_local2(N)
E = sw_diag(H, 1000)
# check the result is diagonal
println(ps.opnorm(E - ps.diag(E)) / ps.opnorm(E))

# but we lose the norm :
println(ps.opnorm(E))
println(ps.opnorm(H))
