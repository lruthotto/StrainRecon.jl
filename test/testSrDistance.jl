using Base.Test
using StrainRecon

@testset "srDistance" begin
(m,n) = (3,2)
M1 = trues(m,n)
M2 = falses(m,n)
w1 = [1.0;0.0]
w2 = [1.0;0.0]
@test abs(srDistance(M1,w1,M2,w2)-sqrt(2*m)) < 1e-12
@test abs(srDistance(M1,w1,M2,w2,p=1) - sqrt(m)) < 1e-12
@test abs(srDistance(M1,w1,M2,w2,p=1,pn=1.0) - m) < 1e-12

M3 = falses(m,n+1)
w3 = [1.0;0.0;0.0]
@test abs(srDistance(M1,w1,M3,w3)-sqrt(2*m)) < 1e-12

w3 = [.5;.5]
M1multi = cat(3,M1,M1)
M2multi = cat(3,M1,M1)
w1multi = cat(2,w1,w1)
w2multi = cat(2,w3,w3)
distances = srDistance(M1multi,w1multi,M2multi,w2multi)
@test abs(distances[2] - sqrt(m/n)) < 1e-12

end
