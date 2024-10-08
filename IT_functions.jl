using ITensors

function printIT(O,s)

    L_tot=length(O)
    Hitensor = ITensor(1.)
    for i = 1:L_tot
        Hitensor *= O[i]
    end
    @disable_warn_order A = Array(Hitensor,s',s)
    R=reshape(A,2^L_tot,2^L_tot)
    #display(R)
    return R
end

function auto_corr_IT(A,B::Vector{Any},N,sites)
    vec = Float64[]
    for i in sites
        push!(vec,real(tr(apply(A,B[i]))/2^N))
    end
    return vec
end
auto_corr_IT(A,B,N) = tr(apply(A,B;cutoff=0))/2^N
