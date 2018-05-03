export integratePosterior, wrapperIntegratePosterior, integratePosteriorParallel, iphelper


"""
function integratePosterior(param,rowfuns;K,wquad)

Compute posterior mean of a separable function

Input:
	1. param   - StrainReconParam
	2. rowfuns - functions phi_1,...,phi_m in an array, where phi_i is a function of the i'th block of the binary matrix and the frequency vector w (the target function whose mean is computed is the sum of these functions)

Keyword arguments:
	3. K	 - number of Monte Carlo points in Omega_W
	4. wquad - number of starting points w in Omega_W used to determine K for method overloading

Output:
	1. postmean - value of the expectation integral

Examples:
	Compute the posterior mean of the frequency vector w:
	rowfuns = [(r,w)->w;repmat([(r,w)->zeros(n,1)],m-1)]

	Compute the posterior mean of the binary matrix M:
	rowfuns = Array{Function,1}()
	for i = 1:m
		z1 = zeros((i-1)*(p-1),n)
		z2 = zeros((m-i)*(p-1),n)
		push!(rowfuns, (r,w)->[z1;r;z2])
	end

    Compute the expected misfit 0.5*dot(wres,wres), wres = gamma^0.5 * (Mw-d)
	rowfuns = Array{Function,1}()
	Rd = diag(param.R)
    for i = 1:m
		iblock = (i-1)*(p-1)+1 : i*(p-1)
        push!(rowfuns, (r,w)->[0.5*norm(Rd[iblock] .* (r*w-d[iblock]))^2])
	end
"""

function integratePosterior(param::StrainReconParam,rowfuns::Vector{Function};K::Int64=10000,wquad=randomw(param.n,K))
	K = size(wquad,2)

	n = param.n
	p = param.p
	d = param.d
	m = param.m
	Rd = diag(param.R)
	lambda = param.lambda

	# output size of rowfuns (assuming each has the same dimensions)
	funsize   = size(rowfuns[1](ones(p-1,n),ones(n,1)))
	funlength = prod(funsize)

	# Monte Carlo points, corresponding integrand values and normalization constants
	integrands = zeros(K,funlength)
	normconsts = zeros(K,1)
	logconsts = zeros(K,1)

	# generate all possible rows of M
	Mrows = getB(param)

	# loop through Monte Carlo points
	for idx1 = 1:K
		w = wquad[:,idx1]
		logposteriorvals = zeros(m,p^n)
		funvals = zeros(m,p^n,funlength)

		# loop through block indices
		for idx2 = 1:m
			blockidx2 = (idx2-1)*(p-1)+1 : idx2*(p-1)
			di = d[blockidx2]
			Wdi = Rd[blockidx2]

			# loop through all possible blocks
			for idx3 = 1:p^n
				blockidx3 = (idx3-1)*(p-1)+1 : idx3*(p-1)
				r = Mrows[blockidx3[1]:blockidx3[end],:]
				logposteriorvals[idx2,idx3] = -1/2 * norm(Wdi.*(r*w-di))^2 - lambda * sum(r)
				funvals[idx2,idx3,:] = rowfuns[idx2](r,w)[:]

			end

		end

		maxlogs = maximum(logposteriorvals,2)
		posteriorvals = exp.(logposteriorvals .- maxlogs)
		funposteriorvals = posteriorvals .* funvals

		a = sum(posteriorvals,2) # size = [m,1]
		b = squeeze(sum(funposteriorvals,2),2) # size = [m,funlength]

		c = zeros(m,funlength)
		for idx2 = 1:m
			c[idx2,:] = prod(a[[1:idx2-1;idx2+1:m]]) * b[idx2,:]
		end

		integrands[idx1,:] = sum(c,1)
		normconsts[idx1] = prod(a)
		logconsts[idx1] = sum(maxlogs)


	end

	maxmaxlog = maximum(logconsts)
	logscaling = logconsts - maxmaxlog
	integrands = integrands .* exp.(logscaling)
	normconsts = normconsts .* exp.(logscaling)

	postmean = sum(integrands,1) / sum(normconsts)
	postmean = reshape(postmean,funsize)

	return postmean

end

"""
	wrapperIntegratePosterior(param;K,wquad)

    Wrapper function for integratePosterior() to compute expected (M,w), misfit, and variance of (M,w)

    Input:
		1. param - StrainReconParam

    Optional arguments:
        2. K 	 - number of Monte Carlo points in Omega_W
		3. wquad - number of starting points w in Omega_W used to determine K for method overloading


    Output:
        1. Mmean  - E(M)
        2. wmean  - E(w)
        3. misfit - E(misfit)
		4. Mvar   - Var(M), i.e., E((M-E(M)).^2)
		5. wvar   - Var(w), i.e., E((w-E(w)).^2)
"""

function wrapperIntegratePosterior(param;K::Int64=10000,wquad=randomw(param.n,K))
	K = size(wquad,2)

	d = param.d
	m = param.m
	n = param.n
	p = param.p
	Rd = diag(param.R)

    # E(w)
    rowfuns = [(r,w)->w;repmat([(r,w)->zeros(n,1)],m-1)]
    wmean = integratePosteriorParallel(param,rowfuns,K=K,wquad=wquad)

	# Var(w)
	rowfuns = [(r,w)->(w-wmean).^2;repmat([(r,w)->zeros(n,1)],m-1)]
	wvar = integratePosteriorParallel(param,rowfuns,K=K,wquad=wquad)

    # E(M)
	mrowfuns = Array{Function,1}()
	for i = 1:m
        z1 = zeros((i-1)*(p-1),n)
        z2 = zeros((m-i)*(p-1),n)
        push!(mrowfuns, (r,w)->[z1;r;z2])
	end
    Mmean = integratePosteriorParallel(param,mrowfuns,K=K,wquad=wquad)

	# Var(M)
	mrowfuns = Array{Function,1}()
	for i = 1:m
        z1 = zeros((i-1)*(p-1),n)
        z2 = zeros((m-i)*(p-1),n)
        push!(mrowfuns, (r,w)->[z1;(r-Mmean[(i-1)*(p-1)+1 : i*(p-1), :]).^2;z2])
	end
    Mvar = integratePosteriorParallel(param,mrowfuns,K=K,wquad=wquad)

    # E(misfit)
    prowfuns = Array{Function,1}()
    for i = 1:m
		iblock = (i-1)*(p-1)+1 : i*(p-1)
        push!(prowfuns, (r,w)->[0.5*norm(Rd[iblock] .* (r*w-d[iblock]))^2])
	end
    misfit = integratePosteriorParallel(param,prowfuns,K=K,wquad=wquad)

    return Mmean, wmean, misfit, Mvar, wvar
end

"""
	function integratePosteriorParallel(param,rowfuns;K,wquad)

    Parallelized version of integratePosterior()

	Input:
		1. param   - StrainReconParam
		2. rowfuns - functions phi_1,...,phi_m in an array, where phi_i is a function of the i'th block of the binary matrix and the frequency vector w (the target function whose mean is computed is the sum of these functions)

	Keyword arguments:
		3. K	 - number of Monte Carlo points in Omega_W
		4. wquad - number of starting points w in Omega_W used to determine K for method overloading

    Output:
        1. postmean - value of expected variable
"""

function integratePosteriorParallel(param::StrainReconParam,rowfuns::Vector{Function};K::Int64=10000,wquad=randomw(param.n,K))
	K = size(wquad,2)

	n = param.n
	p = param.p
	d = param.d
	m = param.m
	Rd = diag(param.R)
	lambda = param.lambda

	# output size of rowfuns (assuming each has the same dimensions)
	funsize   = size(rowfuns[1](ones(p-1,n),ones(n,1)))
	funlength = prod(funsize)

	# Monte Carlo points, corresponding integrand values and normalization constants
	integrands = zeros(K,funlength)
	normconsts = zeros(K,1)
	logconsts = zeros(K,1)

	# generate all possible rows of M
	Mrows = getB(param)

    # update integrand values and normalization constants from MC points
    updateRes = (a,c,maxlogs,idx1)->(integrands[idx1,:] = sum(c,1);normconsts[idx1] = prod(a);logconsts[idx1] = sum(maxlogs))

	# loop through Monte Carlo points
    i=1
    nextidx = () -> (idx1=i; i+=1; idx1)
    @sync begin
        for pwork = workers()
            @async begin
                while true
                    idx1 = nextidx()
                    if isempty(idx1) || idx1 > K
                        break
                    end
                    a,c,maxlogs = remotecall_fetch(iphelper,pwork,m,n,p,d,Rd,rowfuns,Mrows,wquad[:,idx1],lambda,funlength)
                    updateRes(a,c,maxlogs,idx1)
                end
            end
        end
    end

	maxmaxlog = maximum(logconsts)
	logscaling = logconsts - maxmaxlog
	integrands = integrands .* exp.(logscaling)
	normconsts = normconsts .* exp.(logscaling)

	postmean = sum(integrands,1) / sum(normconsts)
	postmean = reshape(postmean,funsize)

	return postmean
end

"""
	function iphelper(m,n,p,d,Rd,rowfuns,Mrows,w,lambda,funlength)

    Helper function for integratePosteriorParallel

    Input:
        1. m 		  - number of SNP sites
        2. n 		  - number of strains
		3. p 		  - number of categories
        4. d 		  - data vector
		5. Rd 		  - diagonal of the inverse square root of the covariance matrix
        6. rowfuns	  - functions phi_1,...,phi_m in an array, where phi_i is a function of the i'th row of the binary matrix and the frequency vector w (the target function whose mean is computed is the sum of these functions)
        7. Mrows 	  - all possible rows of binary matrix
        8. w 		  - initial starting point of weight vector
        9. lambda 	  - sparsity parameter
        10. funlength - output size of rowfuns

	Output:
		1. a	   - sum of posterior values
		2. c	   - product of a and sum of normalized posterior values
		3. maxlogs - maximum of log posterior values to normalize values

"""
function iphelper(m,n,p,d,Rd,rowfuns,Mrows,w,lambda,funlength)

    logposteriorvals = zeros(m,p^n)
    funvals = zeros(m,p^n,funlength)

	# loop through block indices
	for idx2 = 1:m
		blockidx2 = (idx2-1)*(p-1)+1 : idx2*(p-1)
		di = d[blockidx2]
		Wdi = Rd[blockidx2]

		# loop through all possible blocks
		for idx3 = 1:p^n
			blockidx3 = (idx3-1)*(p-1)+1 : idx3*(p-1)
			r = Mrows[blockidx3[1]:blockidx3[end],:]
			logposteriorvals[idx2,idx3] = -1/2 * norm(Wdi.*(r*w-di))^2 - lambda * sum(r)
			funvals[idx2,idx3,:] = rowfuns[idx2](r,w)[:]

		end

	end

    maxlogs = maximum(logposteriorvals,2)
    posteriorvals = exp.(logposteriorvals .- maxlogs)
    funposteriorvals = posteriorvals .* funvals

    a = sum(posteriorvals,2) # size = [m,1]
    b = squeeze(sum(funposteriorvals,2),2) # size = [m,funlength]

    c = zeros(m,funlength)
    for idx2 = 1:m
        c[idx2,:] = prod(a[[1:idx2-1;idx2+1:m]]) * b[idx2,:]
    end

    return a,c,maxlogs
end
