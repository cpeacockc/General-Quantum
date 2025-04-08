#A code to take in strings and generate pauli-based Hamiltonians
using SparseArrays, LinearAlgebra

function pauli_expand(paulistr::Union{Vector{Int64},Vector{Float64}},flag::String)
    I=spdiagm([1,1])
    local X
    local Y
    local Z
    if flag=="spin"
        X=Complex.(sparse([2,1],[1,2],[1/2,1/2]))
        Y=Complex.(sparse([2,1],[1,2],[im/2,-im/2]))
        Z=Complex.(spdiagm([1/2,-1/2])) 
    elseif flag=="pauli"
        X=Complex.(sparse([2,1],[1,2],[1,1]))
        Y=Complex.(sparse([2,1],[1,2],[im,-im]))
        Z=Complex.(spdiagm([1,-1]))
    else
        println("invalid spinflag (options: spin, pauli)") 
    end
    P = X+im*Y
    M = X-im*Y
    N = length(paulistr)
    BigOp=0
    for i in 1:1:N
        n=paulistr[i]
        if n==0
            Op=I
        elseif n==1
            Op=X
        elseif n==2
            Op=Y
        elseif n==3
            Op=Z
        elseif n==4
            Op=P
        elseif n==5
            Op=M
        end

        if i==1
            BigOp=copy(Op)
        else
            BigOp = kron(Op,BigOp)
        end
    end

    #if real(BigOp)==BigOp
    #    BigOp = real(BigOp)
    #end

    return BigOp
end


function pauli_expand_rand(paulistr::Vector{Int64},flag::String,rands)
    I=spdiagm([1,1])
    local X
    local Y
    local Z
    if flag=="spin"
        X=Complex.(sparse([2,1],[1,2],[1/2,1/2]))
        Y=Complex.(sparse([2,1],[1,2],[im/2,-im/2]))
        Z=Complex.(spdiagm([1/2,-1/2])) 
    elseif flag=="pauli"
        X=Complex.(sparse([2,1],[1,2],[1,1]))
        Y=Complex.(sparse([2,1],[1,2],[im,-im]))
        Z=Complex.(spdiagm([1,-1])) 
    end
    P = X+im*Y
    M = X-im*Y
    N = length(paulistr)
    BigOp=0
    k=0
    for i in 1:1:N
        n=paulistr[i]
        if n==0
            Op=I
        elseif n==1
            Op=X
        elseif n==2
            Op=Y
        elseif n==3
            Op=Z
        elseif n==4
            Op=P
        elseif n==5
            Op=M
        elseif n==6
            k+=1
            Op=rands[k,1,:,:]+rands[k,2,:,:] .* im
            Op /= norm(Op)
        end

        if i==1
            BigOp=copy(Op)
        else
            BigOp = kron(Op,BigOp)
        end
    end
    return BigOp
end


function Sz_site_fullOp(N::Int64,site::Int64,flag::String)
    str = zeros(N)
    str[site]=3
    return pauli_expand(str,flag)
end

function H_Ising(N,ZZ_field,X_field,Z_field,flag)

    Big_Op=spzeros(2^N,2^N)

    for i in 1:N-1

        str = zeros(N)
        str[i]=3;str[i+1]=3 #Z_i Z_i+1
        Big_Op += ZZ_field[i] .* pauli_expand(str,flag)
    end


    for i in 1:N
        str = zeros(N)
        str[i]=3;#Z_i
        Big_Op += Z_field[i] .* pauli_expand(str,flag)

        str = zeros(N)
        str[i]=1;#X_i
        Big_Op += X_field[i] .* pauli_expand(str,flag)
    end
    return Big_Op
end



function rand_2site_U(N,site,rands,flag)

        str = zeros(N)
        str[site] = 6 #Rand
        str[site+1] = 6 #Rand
        Big_Op = pauli_expand_rand(str,flag,rands)
    return Big_Op
end

function rand_2site_X(N,site,rand,flag)
    
    Big_Op=spzeros(2^N,2^N)

    str = zeros(N)
    str[site] = 1 #X_i
    str[site+1] = 1 #X_i+1
    Big_Op += rand .* pauli_expand(str,flag)
return Big_Op
end
function rand_2site_XY(N,site,rand,flag)
    
    Big_Op=spzeros(2^N,2^N)

    str = zeros(N)
    str[site] = 1 #X_i
    str[site+1] = 2 #Y_i+1
    Big_Op += rand .* pauli_expand(str,flag)
return Big_Op
end

function rand_2site_Y(N,site,rand,flag)
    
    Big_Op=spzeros(2^N,2^N)

    str = zeros(N)
    str[site] = 2 #Y_i
    str[site+1] = 2 #Y_i+1
    Big_Op += rand .* pauli_expand(str,flag)
return Big_Op
end
function rand_2site_Z(N,site,rand,flag)
    
    Big_Op=spzeros(2^N,2^N)

    str = zeros(N)
    str[site] = 3 #Z_i
    str[site+1] = 3 #Z_i+1
    Big_Op += rand .* pauli_expand(str,flag)
return Big_Op
end




function spin_state(N,flag)
    up=[1, 0]
    dn=[0, 1]
    if flag=="Neel"
        state=copy(up)
        for i in 2:N
            if iseven(i)==true
                state=kron(state,dn)
            else
                state=kron(state,up)
            end
        end
    elseif flag=="Up"
        state=copy(up)
        for i in 2:N
            state=kron(state,up)
        end
    elseif flag=="Dn"
        state=copy(dn)
        for i in 2:N
            state=kron(state,dn)
        end
    elseif flag=="Rand"
        state = rand(2^N)
    end
    return state./norm(state)
end

function spin_densm(N,flag)
    if flag=="Neel"
        state=kron(spin_state(N,"Neel")',spin_state(N,"Neel"))
    elseif flag=="Up"
        state=spin_state(N,"Up")*spin_state(N,"Up")'
    elseif flag=="Dn"
        state=spin_state(N,"Dn")*spin_state(N,"Dn")'
    elseif flag=="Rand"
        Q = Symmetric(rand(Complex{Float64},2^N,2^N))
        state = Q*Q'/tr(Q*Q')
    end
    return state
end

Op_t(t,H) = exp(-im .* Matrix(H) .* t)

function ED_exp(H,state,N,t_array)
    if N>=14
        println("System too large")
    else
        state_array = []
        
        for t in t_array
            push!(state_array,Op_t(t,H)*state)
        end
        return state_array
    end
end

comm(A,B) = A*B - B*A
acomm(A,B) = A*B + B*A


function DM_measure(rho,flag,sites,spinflag)
    N = Int(log2(size(rho)[1]))
    meas_arr = []
    for i in sites
        str = zeros(N)
        if flag == "X"
            str[i]=1
            A = pauli_expand(str,spinflag)
            push!(meas_arr,tr(rho*A))
        elseif flag == "Y"
            str[i]=2
            A = pauli_expand(str,spinflag)
            push!(meas_arr,tr(rho*A))
        elseif flag == "Z"
            str[i]=3
            A = pauli_expand(str,spinflag)
            push!(meas_arr,tr(rho*A))
        elseif flag == "J"
            str[i]=1
            X1 = pauli_expand(str,spinflag)
            str = zeros(N)
            str[i+1]=1
            X2 = pauli_expand(str,spinflag)
            str = zeros(N)
            str[i]=2
            Y1 = pauli_expand(str,spinflag)
            str = zeros(N)
            str[i+1]=2
            Y2 = pauli_expand(str,spinflag)
            J = -2 .* (Y1*X2-X1*Y2)
            push!(meas_arr,tr(rho*J))
        end
    end
    return meas_arr
end

function state_measure(state,op,sites,spinflag)
    N = Int(log2(size(state)[1]))
    meas_arr = []
    if typeof(op)==String
        for i in sites
            str = zeros(N)
            if op == "X"
                str[i]=1
                A = pauli_expand(str,spinflag)
                push!(meas_arr,state'*A*state)
            elseif op == "Y"
                str[i]=2
                A = pauli_expand(str,spinflag)
                push!(meas_arr,state'*A*state)
            elseif op == "Z"
                str[i]=3
                A = pauli_expand(str,spinflag)
                push!(meas_arr,state'*A*state)
            elseif op == "J"
                str[i]=1
                X1 = pauli_expand(str,spinflag)
                str = zeros(N)
                str[i+1]=1
                X2 = pauli_expand(str,spinflag)
                str = zeros(N)
                str[i]=2
                Y1 = pauli_expand(str,spinflag)
                str = zeros(N)
                str[i+1]=2
                Y2 = pauli_expand(str,spinflag)
                J = -2 .*(Y1*X2-X1*Y2)
                push!(meas_arr,state'*J*state)
            else
                println("Incorrect Op string. Try inputting matrix operator")
            end
        end
        return real.(meas_arr)
    else
        return real(state'*op*state)
    end

end


function bitarr_to_int(arr)
    arr = reverse(arr)
    return sum(arr .* (2 .^ collect(length(arr)-1:-1:0)))
end


U_t(t,H) = expt(-im*H*t)


function Schrodinger_evolution(psi_0,H,ttrange)

    state_array=[]
    for t in trange
        psi = U_t(t,H)*psi_0
        push!(psi,state_array)
    end
    return state_array
end





#=
function drho(H,state,N,t_array,A_array,g_array)
    
    drho = -im .* comm(rho,H)
    for i in 1:1:length(g_array)
        A = A_array[i]
        drho += g_array[i] .* (A*rho*A' - 0.5 .* acomm(A'A,rho))
    end
    return drho
end
=#