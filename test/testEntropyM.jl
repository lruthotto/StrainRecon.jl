using StrainRecon
using Base.Test

@testset "entropyM" begin

M  = [1 1 1; 1 1 0; 1 0 1; 1 0 0; 0 1 1; 0 1 0; 0 0 1;]
(m,n) = size(M)
R     = StrainRecon.sdiag(ones(m)*100);
atrue = [.6; 0.3; 0.1]
Data  = M*atrue

param = getStrainReconParam(Data,n,R=R)

entropy = entropyM(param,atrue)

@test entropy >= 0 && entropy <= m*n
end
