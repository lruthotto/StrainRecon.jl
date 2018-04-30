export solveMAP

"""
function solveMAP

Find MAP estimate using Block Coordinate Descent

Input:

	1. param       - StrainReconParam

Keyword arguments:

	2. maxTry      - number of BCD trials for solving min |Mw-d|, each trial start with random w drawn from uniform distribution
	3. storeInterm - return (M,w) with lowest misfit and history of each BCD iterations for (M,w) (storeInterm = 0)
                     return all (M,w) and intermediate history of (M,w) during BCD iteration (1)
                     return all (M,w) but no intermediate history of (M,w) during BCD iteration (2)
	4. wtolBCD     - tolerance for breaking out of BCD iteration when |w_prev - w_current| < wtolBCD
	5. maxIterBCD  - maximum number of iterations per BCD trial
	6. maxIterQP   - maximum number of iterations for findw (used in asqp)
	7. pwr           - exponent of the likelihood l_p norm

Output:

	1. wmap - optimal w (storeInterm = 0)
              all w and intermediate history from all BCD trials, sorted by misfit (1)
              all w but no intermediate history from all BCD trials, sorted by misfit (2)
	2. Mmap - optimal M (storeInterm = 0)
              all M and intermediate history from all BCD trials, sorted by misfit (1)
              all M but no intermediate history from all BCD trials, sorted by misfit (2)
	3. his  - history of misfit, |M_old-M_current|, |w_old-w_current| for optimal(M,w) with lowest misfit (0)
              history of misfit, |M_old-M_current|, |w_old-w_current| for all (M,w) and intermediate history of each BCD iterations from all BCD trials sorted by misfit (1)
              history of misfit, |M_old-M_current|, |w_old-w_current| for all (M,w) and no intermediate history from all BCD trials sorted by misfit (2)

"""

function solveMAP(param::StrainReconParam;maxTry::Int64=10,storeInterm = 0,wtolBCD::Float64=1e-3,maxIterBCD::Int64=10,maxIterQP::Int64=5,pwr::Float64=param.pwr)

	n = param.n

    if storeInterm != 1 # return (M,w) from BCD iterations with lowest misfit
        wsqp = zeros(n,maxTry)
        Msqp = falses(length(param.d),n,maxTry)
        Jsqp = zeros(maxTry)
        his = zeros(maxIterBCD,3,maxTry) # each frame for one iteration of BCD and is maxIterBCD-by-3, total of maxTry frames

        # parallelized version
        # println("parallelized version")
        updateRes = (w,m,hisf,idx) -> (wsqp[:,idx] = w;  Msqp[:,:,idx] = m; Jsqp[idx] = hisf[size(hisf,1),1]; his[1:size(hisf,1),1:size(hisf,2),idx] = hisf)

    else
        # return history of binary matrix and weights from all BCD iterations
        wsqp = zeros(n,maxIterBCD + 1,maxTry) # n-by-intermediate history for each BCD iteration-by-total BCD trials
        Msqp = falses(length(param.d),n, maxIterBCD, maxTry) # m-by-n-by-intermediate history for each BCD iteration-by-total BCD trials
        Jsqp = zeros(maxTry)
        his = zeros(maxIterBCD,3,maxTry) # each frame for one iteration of bcd-by-3, total of maxTry frames

        updateRes = (w,m,hisf,idx) -> (wsqp[1:size(w,1),1:size(w,2),idx] = w;  Msqp[1:size(m,1),1:size(m,2),1:size(m,3),idx] = m; Jsqp[idx] = hisf[size(hisf,1),1]; his[1:size(hisf,1),1:size(hisf,2),idx] = hisf)
    end

    w0   = randomw(n,maxTry)
    i=1
    nextidx = () -> (idx=i; i+=1; idx)

    @sync begin
        for pworkers = workers()
            @async begin
                while true
                    idx = nextidx()
 	 				if isempty(idx) || idx > maxTry
 	 					break
 	 				end
                    # n = 1 case, a0 is always [1]
                    if n == 1
                        w,m,hisf= remotecall_fetch(bcd,pworkers,param,[w0[idx]];wtol=wtolBCD,maxIter=maxIterBCD,maxIterQP=maxIterQP,storeInterm=storeInterm,out=-1,pwr=pwr)
                    else
                        w,m,hisf= remotecall_fetch(bcd,pworkers,param,w0[:,idx];wtol=wtolBCD,maxIter=maxIterBCD,maxIterQP=maxIterQP,storeInterm=storeInterm,out=-1,pwr=pwr)
                    end
                    updateRes(w,m,hisf,idx)
                end
            end
        end
    end

"""
    for k=1:maxTry
	    # asqp,his  = sqp(param, a0[:,k],storeInterm=false,out=-1,atol=atolSQP,maxIter=maxIterSQP, maxIterIPQP=maxIterIPQP)
	    asqp,msqp,his  = bcd(param, a0[:,k],storeInterm=false,out=-1,atol=atolSQP,maxIter=maxIterSQP,pwr=pwr)
	    Asqp[:,k] = asqp
		Msqp[:,:,k]  = msqp
	    Jsqp[k]   = misfit(msqp,asqp,param,pwr=pwr)[1];
	end

"""

	if storeInterm == 1
		id   = sortperm(Jsqp) # vector to sort by smallest to largest
		wmap = wsqp[:,:,id] # n-by-history of each BCD iteration-by-total BCD iterations
		Mmap = Msqp[:,:,:,id] # m-by-n-by-history of each BCD iteration-by-total BCD iterations, needs to be converted to bit array for input to misfit
		his = his[:,:,id] # maxIter-by-3-by-maxTry
		return wmap,Mmap,his # return all (M,w) and intermediate (M,w) during BCD iterations and history for each BCD trial
	elseif storeInterm == 0
		ksqp = indmin(Jsqp)
		wmap = wsqp[:,ksqp]
		Mmap = Msqp[:,:,ksqp] # needs to be converted to bit array for input to misfit
		his = his[:,:,ksqp]
		return wmap,Mmap,his # return (M,w) with lowest misfit and history for each BCD trial
	else
		ksqp = sortperm(Jsqp)
		wmap = wsqp[:,ksqp]
		Mmap = Msqp[:,:,ksqp] # needs to be converted to bit array for input to misfit
		his = his[:,:,ksqp]
		return wmap,Mmap,his # return all (M,w) and history for each BCD trial
	end

end
