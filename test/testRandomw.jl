@testset "randomw" begin
for n=2:6
	println("n=$n")
	A0 = randomw(n,1)
	@test isa(A0,Vector)
	@test abs(sum(A0)-1) < 1e-8
	@test all(diff(A0).<=0)

	nSamp = 100
	A0 = randomw(n,nSamp)
	@test size(A0,2)==nSamp
	@test all(abs.(sum(A0,1)-1) .< 1e-8)
	@test all(diff(A0,1).<=0)
end
end
