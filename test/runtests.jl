using StrainRecon
using Base.Test

@testset "StrainRecon" begin
	println("=== testing StrainRecon.jl code ===")
	println("")
	include("testFindw.jl")
	include("testSolveMAP.jl")
	include("testSolveConvexMAP.jl")
	include("testReducedMisfit.jl")
	include("testRandomw.jl")
	include("testRandomM.jl")
	include("testBCD.jl")
	include("testIntegratePosterior.jl")
	include("testASQP.jl")
	include("testIPLP.jl")
	include("testEntropyM.jl")
	include("testSrDistance.jl")
	include("testIntmat2binmat.jl")
end
