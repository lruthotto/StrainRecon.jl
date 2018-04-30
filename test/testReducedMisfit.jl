
@testset "reducedMisfit" begin
	using jInv.Utils

(m,n) = (40,5)
M     = rand(m,n)
M     = (M.>.3)*1
R     = StrainRecon.sdiag(rand(m))
atrue = randomw(n)
Data  = M*atrue

param = getStrainReconParam(Data,n,R=R)

print("\t--test gradient of reduced misfit--")
function testGradRedMisfit(x,v=[])
    Jc,res,wres,M,dJ,H = reducedMisfit(param,x,doDerivative=true)
	if !(isempty(v))
        return Jc, dot(dJ,v)
	else
        return Jc
	end
end

atest = rand(n); atest /= sum(atest)
chkDer, = checkDerivative(testGradRedMisfit,atest)
@test chkDer

println("\t--test Hessian of reduced misfit--")
function testHessianRedMisfit(x,v=[])
    Jc,res,wres,M,dJ,H = reducedMisfit(param,x,doDerivative=true)
 	if !(isempty(v))
        return vec(dJ), H*v
	else
        return vec(dJ)
	end
end
chkDer, = checkDerivative(testHessianRedMisfit,atest)
@test chkDer
end
