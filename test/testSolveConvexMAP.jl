using StrainRecon
using Base.Test

if StrainRecon.hasGurobi
	@testset "getMAPConvex" begin

	M  = [1 1 1; 1 1 0; 1 0 1; 1 0 0; 0 1 1; 0 1 0; 0 0 1;]
	(m,n) = size(M)
	R     = StrainRecon.sdiag(ones(m)*100);
	atrue = [.6; 0.3; 0.1]
	Data  = M*atrue

	param = getStrainReconParam(Data,n,R=R)

	Mmap,amap = solveConvexMAP(param)
	@test norm(amap - atrue)/norm(atrue) < 1e-8
	@test norm(Mmap*amap-Data) < 1e-7
	end
end
