export findw
"""
function findw(param,M;maxIterQP,out,pwr)

Find a frequency vector w in Omega_W that solves the constrained linear least
squares problem

min_w | M*w - d |_{Gamma,pwr}

Input:

	1. param     - StrainReconParam
	2. M         - binary matrix
    3. maxIterQP - maximum number of iterations for asqp
	4. out       - show intermediate results for asqp/iplp if > 1
	5. pwr       - exponent in the likelihood function

Output:

	1. wsol      - solution vector for constrained least squares problem

"""
function findw(param::StrainReconParam,M::BitArray{2};maxIterQP::Int=5,out=-1,pwr::Float64=param.pwr)

	(m,n) = size(M)
	R     = param.R
	d  = param.d

	# equality constraint
	(Ci,ci,Ce,ce) = getConstraints(param)

	if pwr==2.0
		# solve equality constrained QP problem and see if constraints are satisfied
		Q   = M'*((R'*R)*M) # diagm(collect(0:n-1));
		c   = M'*(R'*(R*d))
		x0  = 1./collect(1:n)
		x0  ./= sum(x0)
		wsol, iter, Wc = asqp(Q,-c,Ce,ce,Ci,ci,x0,maxIter=maxIterQP,out=out)
	elseif pwr==1.0
        # solve equality constrained linear programming with interior point method
        # return wsol, Lagrange multipliers, slack variables, history of iterations
        w0  = 1./collect(1:n)
		w0  ./= sum(w0)
		# set up linear system to be solved by IPLP
	    ni = size(Ci,1)
	    ne = size(Ce,1)
	    Ae = [M -eye(m) eye(m) zeros(m,ni);
	          Ce zeros(ne,2*m) zeros(ne,ni);
	          Ci zeros(ni,2*m) -eye(ni)]
	    ae = [vec(d); vec(ce); vec(ci); ]

	    c  = [zeros(n); diag(R); diag(R); zeros(ni)]
	    # how to initialize Lagrange mutlipliers and slack variables?
		x0 = rand(n+2*m+ni)
		y0 = rand(size(Ae,1))
	    s0 = rand(size(x0))

        wsol, lm, s, his = iplpPathFollowing(Ae,ae,c,x0,y0,s0,maxIter=100,out=-1,ftol=1e-12,mutol=1e-12)
		wsol = wsol[1:n]
	end

	return wsol
end
