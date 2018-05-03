export solveConvexMAP

"""
function solveConvexMAP

Find global MAP estimate using Gurobi

Input:

	1. param - StrainReconParam

Keyword arguments:

	2. pwr - exponent in the likelihood (must be 1.0 or 2.0)
	3. out - output flag for Gurobi solver
	4. tol - MIPGap tolerance parameter for Gurobi solver

Output:

	1. M      - estimated binary matrix
	2. w      - estimated frequency vector
    3. status - return code of the Gurobi solver
"""

function solveConvexMAP(param::StrainReconParam;pwr::Float64=param.pwr,out::Int=0,tol::Float64=1e-6)

    pwr = Int(pwr)

    m = param.m
    n = param.n
    p = param.p
    q = m*(p-1)
    d = param.d
    dRp = diag(param.R).^pwr

    model = Model(solver=GurobiSolver(OutputFlag=out,MIPGap=tol))
    @variable(model, 0 <= w[1:n] <=1)
    @variable(model, 0 <= W[1:q,1:n] <=1)
    @variable(model, 0 <= M[1:q,1:n] <=1,Int)
    if pwr == 1
        @variable(model, zplus[1:q] >= 0)
        @variable(model, zminus[1:q] >= 0)
    end

    @constraint(model,W.<=M)
    for k=1:n
        @constraint(model,W[:,k].<= w[k])
        @constraint(model,W[:,k].>= M[:,k] + w[k]-1)
    end
    for k=1:n-1
        @constraint(model,w[k]>=w[k+1])
    end
    @constraint(model, sum(w)==1)
    if pwr == 1
        @constraint(model, M*w-d .== zminus+zplus)
    end

    if p > 2
        for k=1:n
            for j=1:m
                @constraint(model, sum(M[(j-1)*(p-1)+1 : j*(p-1), k]) <= 1)
            end
        end
    end

    e = ones(n)

    if pwr == 2
        JuMP.setobjective(model,:Min, dRp'*(W*e-d).^2 );
    elseif pwr == 1
        JuMP.setobjective(model,:Min, dRp'*(zplus+zminus) );
    end

    status = solve(model)

    return BitArray(getvalue(M)), getvalue(w), status

end
