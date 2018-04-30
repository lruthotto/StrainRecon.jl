export entropyM

"""
function entropyM(param,w;pwr)

Compute the entropy of the probability p(M|w,d) for given w

Input:

	1. param - StrainReconParam
	2. w     - frequency vector

Keyword argument:

    3. pwr   - exponent in the likelihood function (default = param.pwr)

Output:

	1. E     - entropy
"""

function entropyM(param::StrainReconParam,w::Vector;pwr::Float64=param.pwr)
	n  = length(w)
	d  = param.d
	Rp = diag(param.R.^pwr)
	p  = param.p
	m  = param.m

	B  = getB(param)
	Ba = reshape(B*w,p-1,:)'
	Bs = reshape(param.lambda * vec(sum(B,2)), p-1,:)'

	blockEntropies = zeros(m)

	for k=1:m
		blockidx = (k-1)*(p-1)+1 : k*(p-1)
		Jk = - vec(sum(Rp[blockidx]'.*abs.(Ba.-d[blockidx]').^pwr,2)/pwr - sum(Bs,2))
		Jk-= maximum(Jk)
		Prob = exp.(Jk)
		Prob/=sum(Prob)

		nonzero = Prob .> 0

		blockEntropies[k] = abs(dot(Prob[nonzero], log2.(Prob[nonzero])))
        end

	entropy = sum(blockEntropies)

	return entropy
end
