# A code to generate time evolution under Lindblad equation
using DifferentialEquations,PyPlot
include("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//General_Quantum//Pauli Generator.jl")
let
#For superfluidXXZ

N=L_tot=10
d=0.1
gamma=1.
spinflag="spin"
sites=[1,2,3,4,5,6,7,8,9]
str = zeros(N)
str[1]=4;
AUp1 = pauli_expand(str,spinflag)
str = zeros(N)
str[N]=5;
ADnL = pauli_expand(str,spinflag)


rho_0 = spin_densm(N,"Rand")
J_vec = [1,1,d]
X_field = zeros(N)
Y_field = zeros(N)
Z_field = zeros(N)
H = H_XYZ(N,J_vec,X_field,Y_field,Z_field,spinflag)
A_array=[AUp1,ADnL]
g_array=gamma*ones(length(A_array))


#From diffeq solver
function du(rho,p,t)
    drho = -im .* comm(rho,H)
    for i in eachindex(g_array)
        A = A_array[i]
        drho += g_array[i] .* (A*rho*A' - 0.5 .* acomm(A'*A,rho))
    end
    drho
end

tspan = (0.0,50)

prob = ODEProblem(du,rho_0,tspan)

@time sol = solve(prob,Tsit5())

times = sol.t
rhos = sol.u
J_t = zeros(length(sites),length(times))
for i in eachindex(times)
    J_t[:,i]=real.(DM_measure(rhos[i],"J",sites,spinflag))
end


i=0
for site in sites
    i+=1
    plot(times,J_t[i,:],label="site $site")
end
xlabel("times")
legend()
ylabel("current")
title("Exact Solution N=$N, gamma=$gamma, d=$d")
cd("C://MyDrive//Documents//A-Physics-PhD//Dries-Research//Code//SuperFluid_BH//Outs//L$(L_tot)//")

savefig(replace("Exact_Current_Master_TE_N$(N)_d$(d)_g$(gamma)", "." => "_")*".pdf")
cla()
end