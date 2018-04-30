export findM
"""
function findM(param,w;nsol,pwr)

Find a binary matrix M in Omega_M that solves

min_M  \| M*w - d \|\_{Gamma,pwr} + lambda \| M \|\_0,

Input:

	1. param - StrainReconParam
	2. w     - vector of weights

Keyword arguments:

	3. nsol  - returns nsol matrices corresponding to nsol smallest misfits (default = 1)
	4. pwr   - exponent in the likelihood function (default = param.pwr)

Output:

	1. M     - binary matrix of size (q,n) or an array of size (q,n,nsol)
"""
function findM(param::StrainReconParam,w::Vector;nsol::Int64=1,pwr::Float64=param.pwr)
	Rp = diag(param.R.^pwr)
	d  = param.d
	n  = param.n
	p  = param.p
	m  = param.m

	B  = getB(param)
	Ba = reshape(B*w,p-1,:)'
	Bs = reshape(param.lambda * vec(sum(B,2)), p-1,:)'

	if nsol == 1
		M  = falses(length(d),n)
		for k=1:m
			blockidx = (k-1)*(p-1)+1 : k*(p-1)
			ik = indmin(sum(Rp[blockidx]'.*abs.(Ba.-d[blockidx]').^pwr,2) + sum(Bs,2))
			M[blockidx,:] = B[(ik-1)*(p-1)+1 : ik*(p-1),:]
		end
	else
		M  = falses(length(d),n,nsol)
		for k=1:m
			blockidx = (k-1)*(p-1)+1 : k*(p-1)
			ik = sortperm(vec(sum(Rp[blockidx]'.*abs.(Ba.-d[blockidx]').^pwr,2) + sum(Bs,2)))
			for j=1:nsol
				M[blockidx,:,j] = B[(ik[j]-1)*(p-1)+1 : ik[j]*(p-1),:]
			end
		end
	end

	return M
end
