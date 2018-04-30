export iplpAffineScaling, iplpPathFollowing , iplpwrapper

"""
function F(x,y,s,Ae,ae,c;tau)

Set up KKT conditions for solving

min c'x
s.t. Ax = b, x >= 0
(primal problem)

or equivalently

max b'y
s.t. A'y + s = c, s>= 0
(dual problem)

F(x,y,s) = | Ae*x - ae    | = |0|
           | Ae'y + s - c |   |0|
           | x.*s - tau   |   |0|

Consult Nocedal & Wright "Numerical Optimization" Chapter 14.1 for more details

    Input:
        1. x  - starting point, nx1
        2. y  - Lagrange multipliers, nx1
        3. s  - slack variables, nx1
        4. Ae - matrix of weights encoding equality constraints,mxn
        5. ae - equality constraints to satisfy,nx1
        6. c  - starting vector, nx1
    
    Keyword arguments:
        1. tau - non-negativity constraint such that (x,s) >= 0

    Output:
        1. r1 - Ae*x - ae 
        2. r2 - Ae'y + s - c
        3. r3 - XS - tau 
"""
function F(x,y,s,Ae,ae,c;tau=0.0)
    r1 = Ae*x - ae
    r2 = Ae'*y + s - c
    r3 = x.*s - tau

    return (r1,r2,r3)
end

"""
Split output into components correspond to x variables, lagrange multipliers, slack variables
"""
function splitVec(d,nx,ny,ns)
    return d[1:nx], d[nx+1:nx+ny], d[nx+ny+1:end]
end

"""
function iplpAffineScaling(A,b,c,x,y,s;kwargs...)

Wrapper function for iplp()
"""

iplpAffineScaling(A,b,c,x,y,s;kwargs...) = iplp(A,b,c,x,y,s;kwargs...)

"""
function iplpPathFollowing(A,b,c,x,y,s;kwargs...)

Wrapper function for iplp()
"""

iplpPathFollowing(A,b,c,x,y,s;kwargs...) = iplp(A,b,c,x,y,s;sig=iter->max(min(0.5^iter,0.75),1e-2), kwargs...)

function stepLengthIPLP(x,s,dx,ds;eta=0.999)
    alphax = -1./min(minimum(dx./x),-1)
    alphax = min(1,eta*alphax)

    alphas = -1./min(minimum(ds./s),-1)
    alphas = min(1,eta*alphas)

    return min(alphax,alphas), min(alphax,alphas)
end


function solveBigSystem(x,y,s,A,b,c,r1,r2,r3)
    (m,n) = size(A)
    KKT = [A zeros(m,m) zeros(m,n); zeros(n,n) A' eye(n); diagm(s) zeros(n,m) diagm(x)]
    step = KKT\[r1;r2;r3]

    return splitVec(step,n,m,n)
end

function solveNormEq(x,y,s,A,b,c,r1,r2,r3)
    NormEq = A*sdiag(x./s)*A'
    b      = r1 + A*(x.*(r2-r3./x)./s)

    dy3 =  NormEq\b
    dx3 =  - x.*(r2-r3./x-A'*dy3)./s
    ds3 = - s.*(dx3./x) + r3./x
   return [dx3;dy3;ds3]
end

function solveSymInd(x,y,s,A,b,c,r1,r2,r3)
    m,n = size(A)
    SymInd = [-sdiag(s./x) A'; A zeros(m,m)]
    b = [r2-r3./x;r1]

    xt =  SymInd\b
    dx2 = xt[1:n]
    dy2 = xt[n+1:n+m]
    ds2 = - s.*(dx2./x) + r3./x
    return [dx2;dy2;ds2]
end

"""
function iplp(A,b,c,x,y,s;maxIter,out,eta,ftol,mutol,storeInterm,sig,Jinv)

Interior Point Linear Programming (IPLP) solver

    Input:
        1. A - matrix of weights for encoding equality constraints, mxn
        2. b - equality constraints to satisfy, nx1
        3. c - starting vector, nx1
        4. x - starting point, nx1
        5. y - Lagrange multipliers, nx1
        6. s - slack variables, nx1

    Keyword arguments:
        7. maxIter      - number of iterations for Interior Point method
        8. out          - print intermediate outputs for Interior Point method for linear programming if out = 2, final iteration results if out >= 1 or = 0
        9. eta          - step size for updating search direction
        10. ftol        - tolerance for breaking out of iterations if change in sum of search directions (outputs of F()) < ftol AND duality measure < mutol
        11. mutol       - tolerance for breaking out of iterations if change in sum of norm of search directions (outputs of F()) < ftol AND duality measure < mutol
        12. storeInterm - return intermediate history
        13. sig         - function handle for setting centering parameter; if sigma = 0 then all steps are standard Newton steps
        14. Jinv        - function handle for solving 

                          |0 A' I| |dx| = |- (A'y + s - c)  |
                          |A 0  0| |dy| = |- (Ax - b)       |
                          |S 0  X| |ds| = |-XSe + sigma*mu*e|
                          
                          S = diag(s), X = diag(x), sigma = centering parameter in (0,1), mu = centrality measure = (x.*s)/n (n = length(s) = length(x)), e = unit vector of ones 

    Output:
        1. x   - solution vector
        2. y   - Lagrange multipliers
        3. s   - slack variables
        4. his - history of iterations storing sum of norm of search directions, duality measure, etc
    
"""
function iplp(A,b,c,x::Vector,y::Vector,s::Vector; maxIter::Int=10,out::Int=1,eta::Real=0.999,
                    ftol::Real=1e-4,mutol::Real=1e-4,storeInterm=0,sig::Function=iter->0.0,Jinv::Function=solveBigSystem)


    nx = length(x)
    ny = length(y)
    ns = length(s)

    (f1,f2,f3) = F(x,y,s,A,b,c,tau=sig(1)*dot(x,s)/nx)

    his = zeros(maxIter,8)

    if storeInterm == 1
        X = zeros(nx,maxIter)
        Y = zeros(ny,maxIter)
        S = zeros(ns,maxIter)
    end

    if out==2
        @printf "=== Interior Point for LP ===\n"
        @printf "%4s\t%3s\t\t%2s\t\t%4s\t\t%4s\t\t%4s\t\t%5s\t\t%3s\n" "iter" "|F|" "mu" "|dx|" "|dy|" "|ds|" "alpha" "sig"
    end
    flag = -1
    iter = 1
    for iter=1:maxIter
        (dx,dy,ds) = Jinv(x,y,s,A,b,c,-f1,-f2,-f3)
        aMaxx,aMaxs = stepLengthIPLP(x,s,dx,ds,eta=eta)

        # update
        x += aMaxx*dx
        y += aMaxs*dy
        s += aMaxs*ds

        if storeInterm == 1
            X[:,iter] = x
            Y[:,iter] = y
            S[:,iter] = s
        end

        (f1,f2,f3) = F(x,y,s,A,b,c,tau=sig(iter)*dot(x,s)/nx)
        # his[iter,:] = [norm(f1)+norm(f2)+norm(f3) dot(s,x)/nx norm(dx) norm(dy) norm(ds) aMaxx aMaxs sig(iter)]
        his[iter,1:7] = [norm(f1)+norm(f2)+norm(f3) dot(s,x)/nx norm(dx) norm(dy) norm(ds) aMaxx aMaxs ]

        if out==2
            @printf "%4d\t%1.2e\t%1.2e\t%1.2e\t%1.2e\t%1.2e\t%1.2e\t%1.2e\n" iter his[iter,1] his[iter,2] his[iter,3] his[iter,4] his[iter,5]  his[iter,6] his[iter,8]
        end
        if his[iter,1]<ftol && (his[iter,2]<mutol)
            flag = 0
            break
        end
    end

    if out>=0
        if flag==-1
            @printf "iplp iterated maxIter(=%d) times but reached only |F(x,y,s)|=%1.2e (ftol=%1.2e) and mu=%1.2e (mutol=%1.2e)\n" maxIter his[end,1] ftol his[end,2] mutol
        elseif flag==0 && out>=1
            @printf "iplp achieved ftol=%1.2e and mutol=%1.2e at iteration %d. Returned result satisfies |F(x,y,s)|=%1.2e and mu=%1.2e.\n" ftol  mutol iter his[iter,1] his[iter,2]
        end
    end

    (x,y,s) = (storeInterm==1) ? (X[:,1:iter],Y[:,1:iter],S[:,1:iter]) : (x,y,s)
    return x,y,s,his[1:iter,:]
end
