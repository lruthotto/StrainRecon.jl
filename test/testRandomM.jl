using Base.Test
using StrainRecon

@testset "randomM" begin

# check that the columns are unique
(m,n,N) = (3,2,100)
M = randomM(m,n,N=N)
@test size(M) == (m,n,N)
uniqcols = zeros(Int,N)
for idx = 1:N
	uniqcols[idx] = size(unique(M[:,:,idx],2),2)
end
@test all(uniqcols .== n)

# check that the column sums of each block are at most 1
(m,n,p) = (24,2,3)
M = randomM(m,n,p=p)
@test size(M) == ((p-1)*m,n)
blockcolsums = sum(reshape(M,(p-1,:)),1)
@test maximum(blockcolsums) <= 1

end
