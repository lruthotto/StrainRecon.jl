export findM
"""
function findM(param,w;nsol,pwr)

Find a binary matrix M in Omega_M that solves

min_M  \| M*w - d \|\_{Gamma,pwr} + lambda \| M \|\_0,

Input:

	1. param - StrainReconParam
	2. w     - vector of weights

Keyword arguments:

	3. nsol  - returns nsol matrices (default = 1); if nsol = 2, first matrix has rows with lowest difference |M(i)*w-d(i)| by each row i and is the MAP estimator, second matrix has rows with second lowest difference |M(i)*w-d(i)| by each row i but is NOT the second "best" or most likely matrix
	4. pwr   - exponent in the likelihood function (default = param.pwr)
    5. probM - returns p(M) for nsol matrices corresponding to "confidence"/probability for each row of M, size(m,nsol), (default = false); p(M(i)) = C* exp(-0.5 * (|M(i)*w - d(i)|_R)^2) where R is the diagonal of the inverse square of the noise covariance matrix and C is the normalization constant 1 / (sum_{m}{j}(sum_{2^n}{i} exp(-0.5 * (|B(i)*w - d(j)|_R)^2))), where B is a binary matrix with all possible entries of size(2^n,n)

Output:

	1. M  - binary matrix of size (q,n) or an array of size (q,n,nsol)
    2. pM - matrix of p(M) of size (m,nsol), each column represents "confidence" for the rows of M
"""
function findM(param::StrainReconParam,w::Vector;nsol::Int64=1,pwr::Float64=param.pwr,probM=false)
	Rp = diag(param.R.^pwr)
	d  = param.d
	n  = param.n
	p  = param.p
	m  = param.m

	B  = getB(param)
	Ba = reshape(B*w,p-1,:)'
	Bs = reshape(param.lambda * vec(sum(B,2)), p-1,:)'
    pM = zeros(m,nsol)
    
    if nsol > 2^n
        println("nsol can only be <= 2^n = $(2^n), only the MAP estimator will be returned")
        nsol = 1
    end

	if nsol == 1
		M  = falses(length(d),n)
		for k=1:m
			blockidx = (k-1)*(p-1)+1 : k*(p-1)
            Bdiff = sum(Rp[blockidx]'.*abs.(Ba.-d[blockidx]').^pwr,2) + sum(Bs,2) # |B(ik)*w - d(k)|^2_R, size(2^n,1)
			ik = indmin(Bdiff) # index ik in 1:2^n
            mi = B[(ik-1)*(p-1)+1 : ik*(p-1),:] # row of B with lowest difference B[ik]*w - d[k]
            M[blockidx,:] = mi
            # compute p(M_i)
            # println("ik $ik mi $mi Bdiff $Bdiff)")
            pmi = exp.(-0.5*(Rp[blockidx]'.*abs.(mi*w.-d[blockidx]').^pwr + Bs[ik]))
            # println("pmi $pmi")
            normconst = sum(exp.(-0.5*(Bdiff)))
            # println("normconst $normconst")
            pM[k,nsol] = pmi[1]/normconst
		end
	else
		M  = falses(length(d),n,nsol)
		for k=1:m
			blockidx = (k-1)*(p-1)+1 : k*(p-1)
            Bdiff = vec(sum(Rp[blockidx]'.*abs.(Ba.-d[blockidx]').^pwr,2) + sum(Bs,2))
			ik = sortperm(Bdiff)       
            normconst = sum(exp.(-0.5*(Bdiff))) # normalization constant for each row
			for j=1:nsol
                # nth best row
                mj = B[(ik[j]-1)*(p-1)+1 : ik[j]*(p-1),:]
				M[blockidx,:,j] = mj
                
                # compute p(M_i) for each matrix
                ij = ik[j] 
                # println("ij $ij mj $mj Bdiff $Bdiff)")
                pmj = exp.(-0.5*(Rp[blockidx]'.*abs.(mj*w.-d[blockidx]').^pwr + Bs[ij]))
                # println("j $j pmj $pmj")
                # println("normconst $normconst")
                pM[k,j] = pmj[1]/normconst
                
			end
		end
	end

    if probM
        return M, pM
    else
        return M
    end
end