__precompile__()
module StrainRecon
"""
D = ddx(n), 1D derivative operator
"""
function ddx(n)
	return spdiagm((fill(-1.0,n),fill(1.0,n)),(0,1),n,n+1)
end
export sdiag

"""
S = sdiag(s) builds sparse diagonal matrix
"""
function sdiag(a::Vector)
	n = length(a)
	i = collect(1:n+1) # colptr
	j = collect(1:n)   # rowval
	return SparseMatrixCSC(n,n,i,j,a)
end

# Here go all the packages required
using Combinatorics


include("StrainReconParam.jl")
include("findM.jl")
include("findw.jl")
include("misfit.jl")
include("iplp.jl")
include("bcd.jl")
include("solveMAP.jl")


include("randomw.jl")
include("integratePosterior.jl")
include("asqp.jl")
include("entropyM.jl")
include("srDistance.jl")
include("randomM.jl")
include("intmat2binmat.jl")


hasGurobi=false
try
	using Gurobi
	using MUMPS
	hasGurobi = true
catch
end
if hasGurobi
	include("solveConvexMAP.jl")
end

export warmupStrainRecon
function warmupStrainRecon()
	(m,n) = (8,2)
	M     = rand(Bool,m,n)
	R     = sdiag(ones(m))
	atrue = rand(n); atrue/=sum(atrue)
	atrue = sort(atrue,rev=true)
	Data  = M*atrue

	param = getStrainReconParam(Data,n,R=R,label="init_test")

	println("getReduced Misfit")
	@time reducedMisfit(param,atrue; doDerivative=true)

	println("solveMAP")
    @time solveMAP(param;maxTry=1,storeInterm=false,maxIterQP=1,maxIterBCD=1)

end
end
