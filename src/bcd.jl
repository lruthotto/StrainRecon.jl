export bcd

"""
function bcd(param,w;wtol,maxIter,maxIterQP,storeInterm,out,pwr)

Block coordinate descent method for solving

min_{M,w} || M*w - d ||_{Gamma,pwr} + lambda ||M||_0 s.t. w in Omega_W, M in Omega_M

Input:

    1. param       - StrainRecon param
    2. w           - initial starting point w for BCD

Keyword arguments:

    3. wtol        - tolerance for breaking out of BCD when |w_old - w_current| < wtol
    4. maxIter     - maximum number of BCD trials
    5. maxIterQP   - maximum number of iterations for solving quadratic problem in findw (passed to asqp)
    6. storeInterm - return last (M,w) from each BCD trial (storeInterm = 0 or 2)
                     return all (M,w) from each BCD trial (storeInterm = 1)
    7. out         - print intermediate results of findw (passed to asqp) if out > 1, else print nothing
    8. pwr           - exponent of the likelihood l_p norm

Output:

    1. Mopt - optimal M (storeInterm = 0 or 2) or all intermediate history of M (1)
    2. wopt - optimal w (storeInterm = 0 or 2) or all intermediate history of w (1)
    3. his  - history of misfit, |M_old-M_current|, |w_old-w_current| for optimal (M,w) (storeInterm = 0 or 2)
              history of misfit, |M_old-M_current|, |w_old-w_current| for all intermediate history of (M,w) (1)

"""
function bcd(param,w;wtol::Float64=1e-3, maxIter::Int=10, maxIterQP::Int=5, storeInterm = 0, out::Int=1, pwr::Float64=param.pwr)

	n = length(w)
    if storeInterm > 1
        storeInterm = 0
    end

	if storeInterm == 1
		ws      = zeros(n,maxIter+1)
		ws[:,1] = w
		Ms      = falses(size(param.d,1),n,maxIter)
	end

	his = zeros(maxIter,3)

	if out>1
        @printf "=== Sequential Quadratic Program (n=%d) ===\n" n
        @printf "%4s\t%3s\t\t%6s\t\t%6s\n" "iter" "Jc" "|dM|" "|dW|"
	end

	iter =1; flag = -1
	wOld = copy(w); M = falses(size(param.d,1),n); MOld = copy(M)
	while iter < maxIter

		his[iter,1] = misfit(param,M,w,pwr=pwr)[1]

		M     = findM(param,w,pwr=pwr)

		his[iter,2] = sum(abs.(M-MOld))
		if his[iter,2]==0
			flag=0
			# break
		end

		w =  findw(param,M,maxIterQP=maxIterQP,pwr=pwr)
		# evaluate objective function
		his[iter,3] = norm(w-wOld);

		(out>1) &&	@printf "%d\t%1.2e\t%d\t%1.2e\n" iter his[iter,1] his[iter,2] his[iter,3]

		if his[iter,3] < wtol
			flag = 0
			break
		end

		if storeInterm == 1
			ws[:,iter+1] = w;
			Ms[:,:,iter] = M
		end
		wOld = copy(w)
        MOld = copy(M)
		iter +=1
	end

    # need to fix misleading print statement
	if out>=0
        if flag==-1
            @printf "bcd iterated maxIter(=%d) times but reached only |w - wOld|=%1.2e (wtol=%1.2e)\n" maxIter his[iter,3] wtol
        elseif flag==0 && out>=1
            @printf "bcd achieved |w - wOld| = %1.2e < %1.2e = wtol at iteration %d.\n" his[iter,3] wtol iter
        end
    end


	if storeInterm == 1
		return Ms[:,:,1:iter],ws[:,1:iter+1], his[1:iter,:]
	else
		return M,w, his[1:iter,:]
	end
end
