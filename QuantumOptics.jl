using QuantumOptics

b = SpinBasis(1//2) #Spin 1/2 SpinBasis
N=5
b_atom = SpinBasis(1//2)
b_coll = tensor([b_atom for i=1:N]...)

sm(i) = embed(b_coll, i, sigmam(b_atom))
sp(i) = embed(b_coll, i, sigmap(b_atom))
sz(i) = embed(b_coll, i, sigmaz(b_atom))

H = 0

for i in 1:N-1
end