using PauliStrings, Dictionaries

#First create a function which takes in a vector of parameters and outputs the paulistrings for any body interaction

function Base.:*(o1::Operator, o2::Operator,i::Int,j::Int)

    d = UnorderedDictionary{Tuple{Int, Int}, Complex{Float64}}()
    v = o1.v[i] ⊻ o2.v[j]
    w = o1.w[i] ⊻ o2.w[j]
    c = o1.coef[i] * o2.coef[j] * (-1)^count_ones(o1.v[i] & o2.w[j])
    insert!(d, (v,w), c)
    return op_from_dict(d, o1.N)
end


function op_from_dict(d::UnorderedDictionary{Tuple{Int, Int}, Complex{Float64}}, N::Int)
    o = Operator(N)
    for (v,w) in keys(d)
        push!(o.v, v)
        push!(o.w, w)
        push!(o.coef, d[(v,w)])
    end
    return o
end

function all_k_local(N::Int, k::Int)
    O = Operator(N)
    for i in 0:2^N-1
        for j in 0:2^N-1
            if pauli_weight(i,j)==k
                push!(O.v, i)
                push!(O.w, j)
                push!(O.coef, (1im)^ycount(i, j))
            end
        end
    end
    return O
end

function atmost_k_local(N::Int, k::Int)
    O = Operator(N)
    for i in 0:2^N-1
        for j in 0:2^N-1
            if pauli_weight(i,j)<=k
                push!(O.v, i)
                push!(O.w, j)
                push!(O.coef, (1im)^ycount(i, j))
            end
        end
    end
    return O
end

function pauli_weight(v::Int,w::Int)
    return count_ones(v | w)
end

function all_z(N::Int)
    O = Operator(N)
    for i in 0:2^N-1
        push!(O.v, i)
        push!(O.w, 0)
        push!(O.coef, 1)
    end
    return O
end



