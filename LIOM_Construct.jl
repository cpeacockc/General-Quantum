using LinearAlgebra, Statistics,ProgressBars

function LIOM_avg_construct(bn_tot,x_arr,seig)
    local k=0
    avg_vals=Float64[]
    avg_vecs=[]
    for x in x_arr
        k+=1
        min_eigvals=Float64[]
        LIOM_arr = zeros(Int(x/2),size(bn_tot)[2]) 

        for i in 1:1:size(bn_tot)[2]

            bn_i= bn_tot[:,i][2:end]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)

            M=(transpose(L)*L)[1:x,1:x]
            eigM=eigen(M)

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

            LIOM_arr[:,i]=min_eig_vec[1:2:end]
            push!(min_eigvals,min_eig_val)


            end
        @show x
        avg_min_val=median(min_eigvals)[1]
        avg_LIOM=mean(LIOM_arr,dims=2)[1:end]

        push!(avg_vals,avg_min_val)
        push!(avg_vecs,avg_LIOM)
        end
    return (avg_vals,avg_vecs)
end

function LIOM_log_avg_construct(bn_tot,x_arr,seig)
    local k=0
    avg_vals=Float64[]
    avg_vecs=[]
    for x in x_arr
        k+=1
        min_eigvals=Float64[]
        LIOM_arr = zeros(x,size(bn_tot)[2]) 

        for i in ProgressBar(1:1:size(bn_tot)[2])

            bn_i= bn_tot[:,i][2:end]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)[1:x,1:x]

            M=(transpose(L)*L)
            eigM=eigen(M)

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

            LIOM_arr[:,i]=log.(abs.(min_eig_vec))
            #LIOM_arr[:,i]=min_eig_vec
            push!(min_eigvals,min_eig_val)
            #@show min_eig_vec
            #@show abs.(min_eig_vec)
            #@show log.(abs.(min_eig_vec))


            end
        @show x
        avg_min_val=median(min_eigvals)[1]
        avg_LIOM=mean(LIOM_arr,dims=2)[1:end]

        push!(avg_vals,avg_min_val)
        push!(avg_vecs,avg_LIOM)
        end
    return (avg_vals,avg_vecs)
end

function LIOM_log_avg_construct_L(bn_tot,x_arr,seig)
    local k=0
    avg_vals=Float64[]
    avg_vecs=[]
    for x in x_arr
        k+=1
        min_eigvals=Float64[]
        LIOM_arr = zeros(x,size(bn_tot)[2]) 

        for i in ProgressBar(1:1:size(bn_tot)[2])

            bn_i= bn_tot[:,i][2:end]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)

            eigM=eigen(L[1:x,1:x])

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

            LIOM_arr[:,i]=log.(abs.(min_eig_vec))
            #LIOM_arr[:,i]=min_eig_vec
            push!(min_eigvals,min_eig_val)
            #@show min_eig_vec
            #@show abs.(min_eig_vec)
            #@show log.(abs.(min_eig_vec))


            end
        @show x
        avg_min_val=median(min_eigvals)[1]
        avg_LIOM=mean(LIOM_arr,dims=2)[1:end]

        push!(avg_vals,avg_min_val)
        push!(avg_vecs,avg_LIOM)
        end
    return (avg_vals,avg_vecs)
end
function LIOM_log_avg_construct_EA(bn_tot,x_arr,seig)
    local k=0
    avg_vals=Float64[]
    avg_vecs=[]
    for x in x_arr
        k+=1
        min_eigvals=Float64[]
        LIOM_arr = zeros(x,size(bn_tot)[2]) 

        for i in ProgressBar(1:1:size(bn_tot)[2])

            bn_i= bn_tot[:,i]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)

            M=(transpose(L)*L)[1:x,1:x]
            eigM=eigen(M)

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

            LIOM_arr[:,i]=log.(abs.(min_eig_vec))
            #LIOM_arr[:,i]=min_eig_vec
            push!(min_eigvals,min_eig_val)
            #@show min_eig_vec
            #@show abs.(min_eig_vec)
            #@show log.(abs.(min_eig_vec))


            end
        @show x
        avg_min_val=median(min_eigvals)[1]
        avg_LIOM=mean(LIOM_arr,dims=2)[1:end]

        push!(avg_vals,avg_min_val)
        push!(avg_vecs,avg_LIOM)
        end
    return avg_vecs
end
function LIOM_construct(bn_avg,x_arr,seig)
    local k=0
    min_vals=Float64[]
    min_vecs=[]
    for x in x_arr
        k+=1
        #min_eigvals=Float64[]

            bn_i= bn_avg[2:end]

            L = Tridiagonal(bn_i,zeros(length(bn_i)+1),bn_i)

            M=(transpose(L)*L)[1:x,1:x]
            eigM=eigen(M)

            eig_vals=eigM.values
            eig_vecs=eigM.vectors

            min_eig_val=eig_vals[seig]
            min_eig_vec=eig_vecs[:,seig]

            @show x

        push!(min_vals,min_eig_val)
        push!(min_vecs,min_eig_vec[1:2:end]./norm(min_eig_vec[1:2:end]))
        end
    return (min_vals,min_vecs)
end





