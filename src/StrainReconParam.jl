export StrainReconParam, getB, getStrainReconParam

"""
type StrainReconParam

Data and temporary variables for strain reconstruction problem

Fields:

	1. d::Vector           - measurement vector (SNP data)
	2. m::Int              - number of SNP sites
	3. n::Int              - number of strains to be estimated
	4. p::Int              - number of categories (i.e., possible values for one SNP site)
	5. q::Int              - length of the measurement vector, q = m*(p-1)
	6. pwr                 - exponent in the likelihood function
	7. R                   - Weight matrix for the likelihood as the inverse square root of Gamma
	8. label               - label
	9. lambda              - regularization parameter (usually zero)
	10. B::Array{Bool}     - (p-1)*p^n x n binary matrix of all possible blocks of size (p-1) x n
	11.-14. Ci, ci, Ce, ce - (in)equalities to be used in optimization algorithms
"""
type StrainReconParam
	d::Vector{Float64}
	m::Int
	n::Int
	p::Int
	q::Int
	pwr::Float64
	R::SparseMatrixCSC{Float64}
	label::String
	lambda::Float64
	B::BitArray{2}
	Ci::AbstractArray{Float64}
	ci::Vector{Float64}
	Ce::AbstractArray{Float64}
	ce::Vector{Float64}
end

function Base.display(param::StrainReconParam)
	print("StrainReconParam(")
	(!isempty(param.label)) && print("label=$(param.label), ")
	print("length(d)=$(length(param.d)), ")
	print("nStrains=$(param.n))\n")
	print("nCategories=$(param.p))\n")
	print("nSites=$(param.m))\n")
end

"""
function getStrainReconParam(d,n;p,pwr,R,label,lambda,B,Ci,ci,Ce,ce)

Construct and return a StrainReconParam object.

Input: see StrainReconParam. The weight matrix R can also be given as a vector
(of length q) which is then converted to a diagonal matrix, or as a scalar if
the diagonal is constant.
"""
function getStrainReconParam(d::Vector{Float64},n::Int; p::Int=2, pwr::Float64=2.0,
	                     R=sdiag(ones(size(d,1))),label::String="",lambda::Float64=0.0,
	                     B::BitArray{2}=falses(0,0),
	                     Ci::AbstractArray{Float64}=zeros(0,n),ci::Vector{Float64}=zeros(0),
						 Ce::AbstractArray{Float64}=zeros(0,n),ce::Vector{Float64}=zeros(0))

	if length(d) % (p-1) != 0
		error("the length of data must be divisible by nCategories-1")
	else
		m = Int(length(d)/(p-1))
		q = m*(p-1)
	end

	if length(R) == q
		R = sdiag(R)
	elseif length(R) == 1
		R = R*sdiag(ones(q))
	end

	return StrainReconParam(d,m,n,p,q,pwr,R,label,lambda,B,Ci,ci,Ce,ce)
end

"""
function getConstraints(param::StrainReconParam)

constructs contraint matrices and stores them inside param. Recomputes them only if needed.
"""
function getConstraints(param::StrainReconParam)
	if isempty(param.Ci) || isempty(param.Ce) || isempty(param.ci) || isempty(param.ce) ||
		size(param.Ci,2)!=param.n || size(param.Ce,2)!=param.n

		# equality constraint
		param.Ce = ones(1,param.n)
		param.ce = ones(1)

		# lower bound
        Ci1 = eye(param.n)[end,:]
        ci1 = zeros(param.n)[end]

		# monotonicity
		Ci2 = - full(ddx(param.n-1))
		ci2 = zeros(param.n-1)
		param.Ci = [Ci1';Ci2]
		param.ci = [ci1;ci2]
	end
	return param.Ci, param.ci, param.Ce, param.ce
end



"""
function getB(param::StrainReconParam)

returns (p-1)*p^n x n block binary matrix (block size
is (p-1) x n), corresponding to all possible p-ary
barcodes of length n (all possible blocks of matrix M)

Matrix is only constructed when needed and then stored
in param.B

Input:

	param::StrainReconParam

Output:
	B::Array{Bool} - size(B)=((p-1)*p^param.n,param.n)
"""
function getB(param::StrainReconParam)
	if isempty(param.B) || size(param.B,1)!=(param.p-1)*param.p^param.n || size(param.B,2)!=param.n
		param.B = getB(param.n,param.p)
	end
	return param.B
end

function getB(n::Int,p::Int=2)
	# create all possible rows of n elements with integers 1...p
	intmat = zeros(Int,p^n,n)
	idx = 1
	for i = 0:n-1
		for j = 0:p^n-1
			intmat[idx] = mod(floor(Int,j/p^i),p)+1
			idx += 1
		end
	end

	# convert each row to a binary block with p-1 rows (the last row would be redundant)
	B = falses((p-1)*p^n,n)
	for k=1:size(intmat,1)
		fullblock = falses(p,n)
		fullblock[sub2ind(size(fullblock),intmat[k,:],1:n)] = true
		B[(k-1)*(p-1)+1 : k*(p-1),:] = fullblock[1:p-1,:]
	end

	# arrange so that falses go to the upper left corner
	B = flipdim(flipdim(B,1),2)

	return B
end
