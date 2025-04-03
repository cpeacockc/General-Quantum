using ITensors, LinearAlgebra, HDF5, Random, Statistics, Dates, ProgressBars, CurveFit
include("Pauli Generator.jl")

function IT_XYZ_Jump1L_gates(L,s,d,dt,Jump_ops)
    dOpOps = [] #(op)'op array
    gates = ITensor[]
    for Op in Jump_ops
        push!(dOpOps,apply(dag(swapprime(Op,0,1)),Op))
    end

    for dOpOp in dOpOps
        hRelax = -0.5im*(dOpOp)
        GRelax = exp(-im * dt / 2 * hRelax)
        push!(gates, GRelax)
    end
    
    #XYZ clean gates
    for j in 1:(L-1)         
        s1 = s[j]
        s2 = s[j + 1]
        hj =
          ((d[j]) * op("Sz", s1) * op("Sz", s2) +
          (1/2)*(op("S+", s1) * op("S-", s2) +
          op("S-", s1) * op("S+", s2)))
                    
        Gj = exp(-im * dt / 2 * hj)
        push!(gates, Gj)
    end #for
  
    # Include gates in reverse order too
    # (L,L-1),(L-1,L-2),...
  
    append!(gates, reverse(gates))

    return gates

end

function J_calculate(psi,Js)
    J_vals=Float64[]
    for i in eachindex(Js)
        J_calc=inner(psi',Js[i],psi)
        if imag(J_calc)>=1e-12
            error("J is imaginary: J=$(J)")
        end
        push!(J_vals,real(J_calc))
    end
    return J_vals
end


function Jump_calc(prob,Jump_op,psi,ITcutoff,count,totcount)
    if rand(1)[1]<prob
        psi = apply(Jump_op,psi;cutoff=ITcutoff)
        #println("jump!")
        count+=1
        totcount+=1
    end
    return psi,count,totcount
end

function IT_state(state,s)
    if state=="Neel"
        psi = productMPS(s, n -> isodd(n) ? "Up" : "Dn")
      elseif state == "Up"
        psi = productMPS(s, "Up")
      elseif state == "Dn"
        psi = productMPS(s, "Dn")
      elseif state=="Rand"
        psi = randomMPS(s, L) #Psi to be time evolved
      else
        print("incorrect state option")
    end
    return psi
end


function IT_Jump_TE(psi,s,L,gates,Jump_ops,t_list,tau,t_save_index,ITcutoff,gamma,d,k,path,psi_save)

    tN = length(t_list)
    ta=Dates.now()

    mkpath(path)
    cd(path)

    times = Float64[]
iter_runtime = Float64[]
tot_runtime = Float64[]
XArray_t = Array{Float64}[]
YArray_t = Array{Float64}[]
ZArray_t = Array{Float64}[]
JArray_t = Array{Float64}[]
bonddims = Array{Float64}[]

Js = IT_Js(L,s)


totcount=0
tb = Dates.now()
#Start time evolution
@time for i in ProgressBar(1:1:tN)
    tc = Dates.now()
    t = t_list[i]
  
    tb = Dates.now()
    count=0
    #probs=[]
    #Probabilities for Jump
    #
    for Jump_op in Jump_ops
        prob = tau*norm(apply(Jump_op,psi;cutoff=ITcutoff))^2
        psi,count,totcount = Jump_calc(prob,Jump_op,psi,ITcutoff,count,totcount)
    end

    #
    #=
    p1 = tau*norm(apply(C1p,psi;cutoff=ITcutoff))^2
    pL = tau*norm(apply(CLm,psi;cutoff=ITcutoff))^2
 
#   #Jump at site 1 
    if rand(1)[1]<p1
      psi = apply(C1p,psi;cutoff=ITcutoff)
      count+=1
      global totcount+=1
    end

    #jump at site L
    if rand(1)[1]<pL
      psi = apply(CLm,psi;cutoff=ITcutoff)
      
      count+=1
      global totcount+=1
    end

    =#

#   #if no jump, apply gates
    if count==0
    psi = apply(gates, psi;cutoff=ITcutoff)
    
    end

    normalize!(psi)

    #Full if statement is just to save at some checkpoints
    if i in t_save_index || (tc-ta).value == Maxtime
          
      X = expect(psi, "Sx")
      Z = expect(psi,"Sz")
      Y = real((expect(psi,"S+").-expect(psi,"S-"))./2im)
      J=[]
      for i in eachindex(Js)
        J_calc = inner(psi',Js[i],psi)
        if imag(J_calc)>=1e-12
          println("J is imaginary")
        end
        push!(J,real(J_calc))
      end

      push!(XArray_t,X)
      push!(YArray_t,Y)
      push!(ZArray_t,Z)
      push!(JArray_t,J)
      push!(bonddims,linkdims(psi)) 
      push!(times,t)
      push!(iter_runtime,(tc-tb).value)
      push!(tot_runtime,(tc-ta).value)
      
      fid = h5open("IT-k$k.h5", "w")
      create_group(fid, "SF_TE")
      g = fid["SF_TE"]
      g["Val_Array"] = "L = $L ; J=$J; d = $d ; dt = $dt ; ITcutoff=$ITcutoff; ttotal = $ttotal ;  iter = $k; t_list = $t_list"
      g["L"]=L
      g["d"]=d
      g["gamma"]=gamma
      g["XArray_t"] = reshape(reduce(hcat,XArray_t),L,:)
      g["YArray_t"] = reshape(reduce(hcat,YArray_t),L,:)
      g["ZArray_t"] = reshape(reduce(hcat,ZArray_t),L,:)
      g["JArray_t"] = reshape(reduce(hcat,JArray_t),L-1,:)
      g["times"] = times
      #g["iter_runtime"] = iter_runtime
      g["bonddims"] = reshape(reduce(hcat,bonddims),L-1,:)
      #g["t_save_index"] = t_save_index
      g["tot_runtime"] = tot_runtime
      g["ZZ_corr"] = correlation_matrix(psi,"Sz","Sz")
      if psi_save == true
          g["psi"] = psi
      else
      end

      close(fid)
      
      #@show t, totcount
      else
      
      end#if

    
    t≈ttotal && break
    
    end #for
    return psi
end #function 



function create_jump_ops(L,gamma,jumps,spinflag)
    #input tuple of "jumps" of (sites,ops): sites at which jump operators are placed, and op_numbers (X=1,Y=2,Z=3,P=4,M=5)
    Jump_ops = []

    for jump in jumps
        site=jump[1]
        op = jump[2]
        paulistr = zeros(L)
        paulistr[site] = op
        push!(Jump_ops,sqrt(gamma)*pauli_expand(paulistr,spinflag))

    end
    return Jump_ops
end

function H_XYZ_jumps(L,d,spinflag,Jump_ops)
    Jvec=[1,1,d]
    Xvec=zeros(L)
    Yvec = zeros(L)
    Zvec=zeros(L)
    H_0 = H_XYZ(L,Jvec,Xvec,Yvec,Zvec,spinflag)

    H_1=spzeros(2^L,2^L)

    for i in eachindex(Jump_ops)
        H_1 -=  (0.5)*im*Jump_ops[i]' * Jump_ops[i]
    end

    H = H_0+H_1
    return H
end

function Jump_TE(psi_0,L,H,Jump_ops,t_list,dt,t_save_index,gamma,d,k,path,psi_save,spinflag)

    tN=length(t_list)
    ta=Dates.now()
    sites = collect(1:1:L-1)

    mkpath(path)
    cd(path)

    times = Float64[]
    iter_runtime = Float64[]
    tot_runtime = Float64[]
    XArray_t = Array{Float64}[]
    YArray_t = Array{Float64}[]
    ZArray_t = Array{Float64}[]
    JArray_t = Array{Float64}[]
    #bonddims = Array{Float64}[]

    #J_arr=zeros(length(sites),length(times))
    #M_arr=zeros(L,length(times))
    psi=copy(psi_0)

    Probs = zeros(2)
    prob_up=Float64[]
    prob_dn=Float64[]


    tb = Dates.now()
    #Start time evolution
    @time for i in ProgressBar(1:1:tN)
        tc = Dates.now()
        t = t_list[i]
  
        tb = Dates.now()

        for j in eachindex(Jump_ops)
            Probs[j]=dt*state_measure(psi,Jump_ops[j]' * Jump_ops[j],1,spinflag)
        end 
        push!(prob_up,Probs[1])
        push!(prob_dn,Probs[2])
        jump_count=0
        for j in eachindex(Jump_ops)
            if rand(1)[1]<real(Probs[j])
                    
                psi = Jump_ops[j]*psi
                psi = psi/norm(psi)
                jump_count+=1
                #println("Jump!")
            end
        end
        if jump_count==0
            psi = (sparse(I,2^L,2^L)-im*H*dt)*psi
            psi = psi/norm(psi)
        end

        

        if i in t_save_index || (tc-ta).value == Maxtime
          
            J = state_measure(psi,"J",sites,spinflag)
            Z = state_measure(psi,"Z",collect(1:1:L),spinflag)
            X = state_measure(psi,"X",collect(1:1:L),spinflag)
            Y = state_measure(psi,"Y",collect(1:1:L),spinflag)
      
            push!(XArray_t,X)
            push!(YArray_t,Y)
            push!(ZArray_t,Z)
            push!(JArray_t,J)
            
            push!(times,t)
            push!(iter_runtime,(tc-tb).value)
            push!(tot_runtime,(tc-ta).value)
            
            fid = h5open("k$k.h5", "w")
            create_group(fid, "SF_TE")
            g = fid["SF_TE"]
            g["Val_Array"] = "L = $L ; J=$J; d = $d ; dt = $dt ; ttotal = $ttotal ;  iter = $k; t_list = $t_list"
            g["L"]=L
            g["d"]=d
            g["gamma"]=gamma
            g["XArray_t"] = reshape(reduce(hcat,XArray_t),L,:)
            g["YArray_t"] = reshape(reduce(hcat,YArray_t),L,:)
            g["ZArray_t"] = reshape(reduce(hcat,ZArray_t),L,:)
            g["JArray_t"] = reshape(reduce(hcat,JArray_t),L-1,:)
            g["times"] = times
            #g["iter_runtime"] = iter_runtime
            #g["t_save_index"] = t_save_index
            g["tot_runtime"] = tot_runtime
            if psi_save == true
                g["psi"] = psi
            else
            end
      
            close(fid)
            
            #@show t, totcount
            else
            
            end#if
      
          
          t≈ttotal && break
          
          end #for

    return psi
end


function get_data(L,d,gamma,state,data_flag)

    if data_flag == "Sparse"
        cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//Exports")
        fid = h5open("L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state*".h5","r")
        
    elseif data_flag=="IT"
        cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//IT_Exports")
        fid = h5open("IT_L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state*".h5","r")

    elseif data_flag == "Lindblad"
        cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//LindbladApproach//")
        fid = h5open("L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state*".h5", "r")

    end

    g = fid["SF_TE"]
    J_t = read(g,"J_t")
    M_t = read(g,"M_t")
    times = read(g,"times")
    #rhos = read(g,"rhos")
    close(fid)

    return J_t,M_t,times
end

function IT_Js(L::Int64,s)
    Js=[]

    for i in 1:L-1
    os = OpSum()
        os .+= (-1, "Sy",i,"Sx",(i+1))
        os .+= (1, "Sx",i,"Sy",(i+1))
        J_op = MPO(os,s)
        push!(Js,J_op)
    end
    return Js
end


function IT_Js_ring(L::Int64,s)
    Js=[]

    for i in 0:L-1

    os = OpSum()
        os .+= (-1, "Sy",i+1,"Sx",(i+1)%L+1)
        os .+= (1, "Sx",i+1,"Sy",(i+1)%L+1)
        J_op = MPO(os,s)
        push!(Js,J_op)
    end
    return Js
end

function Wavefunction_Export(L,gamma_list,d,K,state,export_flag)

    for gamma in gamma_list    
        #mkpath("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//WaveFunctionApproach//"*"L$(L)-"*replace("d$(d)-", "." => "_")*state*"//L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state)
        if export_flag=="Sparse"
            cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//WaveFunctionApproach//"*"L$(L)-"*replace("d$(d)-", "." => "_")*state*"//L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state)
            path = pwd()
        elseif export_flag == "IT"
            cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//IT_WaveFunctionApproach//"*"IT_L$(L)-"*replace("d$(d)-", "." => "_")*state*"//IT_L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state)
            path = pwd()
        elseif export_flag=="HPC_IT"
            cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//HPC_outs//IT_SF_BH//"*"IT_L$(L)-"*replace("d$(d)-", "." => "_")*state*"//IT_L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state)
            path = pwd()
        elseif export_flag=="HPC_Sparse"
            cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//HPC_outs//SF_BH//"*"L$(L)-"*replace("d$(d)-", "." => "_")*state*"//L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state)
            path = pwd()
        else
            println("Invalid export_flag")
        end



        fid = h5open(readdir(path)[1], "r")
        g = fid["SF_TE"]
        t_list = read(g,"times")
        close(fid)

        J_Array = zeros(L-1,length(t_list),K)
        Z_Array = zeros(L,length(t_list),K)
        psi_Array = complex(zeros(2^L,K))
        times_Array = zeros(length(t_list),K)
        bonddims_Array = zeros(L-1,length(t_list),K)
        #iter_runtime_Array = zeros(length(t_list),K)


        if export_flag == "IT" || export_flag == "HPC_IT"


            local i=0
            for file in ProgressBar(readdir(path)[1:K])
                i+=1
                fid = h5open(file, "r")
                g = fid["SF_TE"]
                times_Array[:,i] = read(g,"times")
                J_Array[:,:,i] = read(g,"JArray_t")
                #bonddims_Array[:,:,i] = read(g,"bonddims")
                #iter_runtime_Array[:,i] = read(g,"iter_runtime")
                Z_Array[:,:,i] = read(g,"ZArray_t")
                close(fid)
            end

            cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//IT_Exports")

            fid = h5open("IT_L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state*".h5","w")
                create_group(fid, "SF_TE")
                g = fid["SF_TE"]
                g["J_array"] = J_Array
                #g["iter_runtime_array"] = iter_runtime_Array
                g["times_array"] = times_Array
                g["times"] = mean(times_Array,dims=2)[:,1]
                g["J_t"] = mean(J_Array,dims=3)[:,:,1]
                g["M_t"] = mean(Z_Array,dims=3)[:,:,1]
                g["K"] = K
                #g["rhos_avg"] = mean(rhos_export,dims=1)[1,:,:]
                #g["iter_runtime_array_avg"] = mean(iter_runtime_Array,dims=2)[:,1]
                #g["bonddims_array"] = bonddims_Array
                #g["bonddims_array_avg"] = mean(bonddims_Array,dims=3)[:,:,1]
            close(fid)

        elseif export_flag == "Sparse" || export_flag == "HPC_Sparse"
            local i=0
            for file in ProgressBar(readdir(path)[1:K])
                i+=1
                fid = h5open(file, "r")
                g = fid["SF_TE"]
                times_Array[:,i] = read(g,"times")
                J_Array[:,:,i] = read(g,"JArray_t")
                #iter_runtime_Array[:,i] = read(g,"iter_runtime")
                Z_Array[:,:,i] = read(g,"ZArray_t")
                close(fid)
            end

            cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//Exports")

            fid = h5open("L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state*".h5","w")
                create_group(fid, "SF_TE")
                g = fid["SF_TE"]
                g["J_array"] = J_Array
                #g["iter_runtime_array"] = iter_runtime_Array
                g["times_array"] = times_Array
                g["times"] = mean(times_Array,dims=2)[:,1]
                g["J_t"] = mean(J_Array,dims=3)[:,:,1]
                g["M_t"] = mean(Z_Array,dims=3)[:,:,1]
                g["K"] = K
                #g["rhos_avg"] = mean(rhos_export,dims=1)[1,:,:]
                #g["iter_runtime_array_avg"] = mean(iter_runtime_Array,dims=2)[:,1]
                #g["bonddims_array"] = bonddims_Array
                #g["bonddims_array_avg"] = mean(bonddims_Array,dims=3)[:,:,1]
            close(fid)

        end
end

    #rhos_export = complex(zeros(K,2^L,2^L))
    #for i in 1:1:K
    #    rhos_export[i,:,:] =psi_Array[:,i]*psi_Array[:,i]'
    #end
end

function Dissipative_Hawking_Export(L,gamma_list,d1,d2,K,state)

    for gamma in gamma_list    
        #mkpath("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//WaveFunctionApproach//"*"L$(L)-"*replace("d$(d)-", "." => "_")*state*"//L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state)
        cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//HawkingRad_XXZ//Outs//Dissipative_TE//"*"IT_L$(L)-"*replace("d1$(d1)-d2$(d2)-", "." => "_")*state*"//IT_L$(L)-"*replace("g$gamma-", "." => "_")*state)
        path = pwd()



        fid = h5open(readdir(path)[1], "r")
        g = fid["SF_TE"]
        t_list = read(g,"times")
        close(fid)

        ZZ_corr_Array = zeros(L,L,K)
        J_Array = zeros(L-1,length(t_list),K)
        Z_Array = zeros(L,length(t_list),K)
        psi_Array = complex(zeros(2^L,K))
        times_Array = zeros(length(t_list),K)
        bonddims_Array = zeros(L-1,length(t_list),K)
        #iter_runtime_Array = zeros(length(t_list),K)

            local i=0
            for file in ProgressBar(readdir(path)[1:K])
                i+=1
                fid = h5open(file, "r")
                g = fid["SF_TE"]
                times_Array[:,i] = read(g,"times")
                J_Array[:,:,i] = real.(read(g,"JArray_t"))
                ZZ_corr_Array[:,:,i] = real.(read(g,"ZZ_corr"))
                #bonddims_Array[:,:,i] = read(g,"bonddims")
                #iter_runtime_Array[:,i] = read(g,"iter_runtime")
                Z_Array[:,:,i] = read(g,"ZArray_t")
                close(fid)
            end

            mkpath("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//HawkingRad_XXZ//Outs//IT_Exports")
            cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//HawkingRad_XXZ//Outs//IT_Exports")

            fid = h5open("IT_L$(L)-"*replace("g$gamma-d1$(d1)-d2$(d2)-", "." => "_")*state*".h5","w")
                create_group(fid, "SF_TE")
                g = fid["SF_TE"]
                g["J_array"] = J_Array
                #g["iter_runtime_array"] = iter_runtime_Array
                g["times_array"] = times_Array
                g["times"] = mean(times_Array,dims=2)[:,1]
                g["J_t"] = mean(J_Array,dims=3)[:,:,1]
                g["M_t"] = mean(Z_Array,dims=3)[:,:,1]
                g["ZZ_corr"] = mean(ZZ_corr_Array,dims=3)[:,:,1]
                g["K"] = K
                #g["rhos_avg"] = mean(rhos_export,dims=1)[1,:,:]
                #g["iter_runtime_array_avg"] = mean(iter_runtime_Array,dims=2)[:,1]
                #g["bonddims_array"] = bonddims_Array
                #g["bonddims_array_avg"] = mean(bonddims_Array,dims=3)[:,:,1]
            close(fid)
        end

end


function SS_calc(J_t,times,div_N)
    t_N = length(times)
    fit_times = times[end-Int(round(t_N/div_N)):end]
    fit_J = J_t[end-Int(round(t_N/div_N)):end]
    fit=poly_fit(fit_times,fit_J, 0)
    return fit[1]
end

function SS_vs_gamma(L,d,state,div_N,gamma_list)
    SS_J = []
    for gamma in gamma_list


        if L>=16
            cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//IT_Exports")
            fid = h5open("IT_L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state*".h5","r")

        else
            cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//Exports")
            fid = h5open("L$(L)-"*replace("g$gamma-d$(d)-", "." => "_")*state*".h5","r")
        end

        g = fid["SF_TE"]
        J_t = read(g,"J_t")
        times = read(g,"times")
        close(fid)

        push!(SS_J,SS_calc(J_t,times,div_N))
    end
    return SS_J
end

