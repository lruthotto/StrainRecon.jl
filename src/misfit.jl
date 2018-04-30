export misfit, reducedMisfit
"""
function misfit(param,M,w;doDerivative,pwr)

Compute the following quantities

misfit:             1/pwr * || R*(M*w - d) ||_pwr^pwr
residual:           M*w - d
weighted residual:  R*(M*w - d)

and optionally the gradient and Hessian of the misfit with respect to w.

Here, R is the inverse square root of the covariance matrix Gamma.

If M is a 3D array, all values are computed separately for each binary matrix
stored in M.

Inputs:

	1. param  - StrainReconParam
	2. M      - binary matrix or a 3D binary array
	3. w      - vector of weights

Keyword arguments:

    4. doDerivative   - flag for computing derivatives (default = false)
    5. pwr            - exponent in the likelihood function (default = param.pwr)

Output:
	1. Jc             - misfit
	2. res            - residual
	3. wres           - weighted residual
	4. dJ             - gradient of misfit w.r.t. w (if doDerivative = true)
	5. H              - Hessian of misfit w.r.t. w (if doDerivative = true)
"""
function misfit(param::StrainReconParam,M::BitArray{2},w::Vector{Float64}; doDerivative::Bool=false,pwr::Float64=param.pwr)
	d    = param.d
	R    = param.R

	res  = M*w-d
	wres = R*res
	Jc   = 1/pwr * sum(abs.(wres).^pwr)

	if doDerivative
		if pwr != 2.0
		    error("derivative for pwr!=2 not implemented");
		end
		dJ = M'*(R'*wres)
		H  = M'*((R'*R)*M)
		return Jc, res, wres, dJ, H
	else
		return Jc, res, wres
	end
end

function misfit(param::StrainReconParam,M::BitArray{3},w::Vector{Float64}; doDerivative::Bool=false,pwr::Float64=param.pwr)
	if doDerivative
		error("derivative for multiple binary matrices not implemented")
	end

	d    = param.d
	n    = param.n
	nM   = size(M,3)
	m    = length(d)

	# allocate space
	Jc   = zeros(nM)
	res  = zeros(m,nM)
	wres = zeros(m,nM)

	# compute misfit and residuals for each M
	for k=1:nM
		Jc[k],res[:,k],wres[:,k] = misfit(param,M[:,:,k],w,pwr=pwr)
	end
	return Jc, res, wres
end

"""
function reducedMisfit(param,w;nsol,doDerivative,pwr)

Compute reduced misfit

min_M 1/pwr * || R*(M*w - d) ||_pwr^pwr

where the minimum is taken over all binary matrices M in Omega_M.

If nsol>1, then nsol matrices, corresponding to nsol smallest misfit values, are
considered and the output is similar to misfit with 3D input array.

Input:

	1. param  - StrainReconParam
	2. w      - vector of weights

Keyword arguments:

	3. nsol           - consider nsol binary matrices (default = 1)
    4. doDerivative   - flag for computing derivatives (default = false)
	5. p              - exponent in the likelihood function (default = param.pwr)

Output:
	1. Jc             - misfit
	2. res            - residual
	3. wres           - weighted residual
	4. M              - the minimizing matrix (or a 3D array if nsol>1)
	5. dJ             - gradient of misfit w.r.t. w (if doDerivative = true)
	6. H              - Hessian of misfit w.r.t. w (if doDerivative = true)

"""
function reducedMisfit(param::StrainReconParam,w::Vector{Float64};nsol::Int=1,doDerivative::Bool=false,pwr::Float64=param.pwr)
	M = findM(param,w,nsol=nsol,pwr=pwr)
	if !doDerivative
		Jc, res, wres = misfit(param,M,w,pwr=pwr)
		return Jc, res, wres, M
	else
		Jc, res, wres, dJ, H = misfit(param,M,w,doDerivative=doDerivative,pwr=pwr)
		return Jc, res, wres, M, dJ, H
	end
end
