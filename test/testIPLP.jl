using StrainRecon
using Base.Test
@testset "IPLP" begin

(m,n) = (28,3)
M = rand(m,n)
M = (M.>.3)*1
a = randomw(n)
Data = M*a
Gamma = (0.001/norm(Data))*eye(m,m)

R  = sparse(inv(sqrt.(Gamma)))
# Gamma = (0.001/norm(Data))*sdiag(ones(m))
# Dobs,  = sampleGaussian(Data,Gamma)
# Dobs  = Data  + .01*randn(size(Data))/norm(Data)
Dobs = Data
param = getStrainReconParam(Dobs,3,R=R,pwr=1.0);
Ci,ci,Ce,ce = StrainRecon.getConstraints(param)

ni = length(ci)
ne = length(ce)
Ae = [M -eye(m) eye(m) zeros(m,ni); Ce zeros(size(Ce,1),2*m) zeros(ne,ni); Ci zeros(ni,2*m) -eye(ni)]
ae = [vec(Dobs); vec(ce); zeros(ni)]

c  = [zeros(n); diag(R); diag(R); zeros(ni)]

x0 = rand(n+2*m+ni)
y0 = rand(size(Ae,1))
s0 = rand(size(x0))

xip,yip,sip,his = iplpPathFollowing(Ae,ae,c,x0,y0,s0,maxIter=100,ftol=1e-10,mutol=1e-10)
wip = xip[1:n]
zpip = xip[n+1:n+m]
zmip = xip[n+m+1:n+2*m]

@test abs(sum(wip)- 1.0) < 1e-10
@test all(diff(wip) .<= 0)
@test all(zpip.>=0)
@test all(zmip.>=0)
@test norm(M*wip - zpip + zmip  - Dobs) < 1e-10
@test norm(M*wip-Dobs)/norm(Dobs) < 1e-10
end
