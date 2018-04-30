using StrainRecon
using Base.Test

@testset "intmat2binmat" begin

Mint = [1 2 3; 2 3 4]
Mbin = intmat2binmat(Mint)
Mtest = [0 1 0; 0 0 1; 0 0 0; 1 0 0; 0 1 0; 0 0 1]
@test Mbin == convert(BitArray{2}, Mtest)

Mbin5 = intmat2binmat(Mint; p=5)
Mtest5 = [0 1 0; 0 0 1; 0 0 0; 0 0 0; 1 0 0; 0 1 0; 0 0 1; 0 0 0]

@test Mbin5 == convert(BitArray{2}, Mtest5)
end
