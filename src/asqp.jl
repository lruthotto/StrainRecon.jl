export asqp, solveKKT, getLinearIndependentConstraints

"""
function asqp(G,c,Ae,be,Ai,bi,x=0*c; maxIter, tolP, out)

Find optimizer for quadratic programming problem using active set method

min_x q(x) = 0.5\*x'\*G*x + c'\*x
subject to Ax >= b

HERE: Assume that active constraints are linearly independent.

	Input:
        1. G 	 - matrix of weights for x, nxn
        2. c 	 - linear vector of weights, nx1
        3. Ai/Ae - matrix of inequality/equality constraints, mxn or pxn
        4. bi/be - inequality/equality constraints to satisfy, mx1 or px1
        5. x 	 - initial starting point if provided or default to origin [0,0,...,0], nx1

	Keyword arguments:
        6. maxIter - number of iterations for finding feasible set and direction before ending asqp
        7. tolP    - tolerance for ending search for direction and asqp
        8. out	   - print intermediate outputs if out > 1

    Output:
        1. x 	- minimizer for quadratic programming problem, nx1
        2. iter - number of iterations to find minimizer
        3. Wci  - inequality working conditions at minimizer, m'xn

"""
function asqp(G,c,Ae,be,Ai,bi,xc=0*c; maxIter::Int64=5, tolP::Float64=1e-10,out::Int64 = 2)

	# if rank(G) < size(G,2)
	# 	Ginv = x->(G+1e-6*eye(size(G,2)))\x
	# else
	# 	Ginv = x->G\x
	#  end

    # println("########## ASQP start ########### \n Ae $Ae \n be $be \n Ai $Ai \n bi $bi \n G $G \n c $c \n x0 $xc")
    Wci = find(Ai*xc-bi .< 0)
	if (length(Wci)>0 && size(Ae,1)>0)  || length(Wci)>1
    	Wci = getLinearIndependentConstraints([Ae;Ai],[vec(collect(1:size(Ae,1)));  size(Ae,1)+Wci])
		Wci = Wci[size(Ae,1)+1:end]-size(Ae,1)
	end
    Aic = Ai[Wci,:]
    # println("Aic $Aic")

    iter = 0
    while true
        if out>1
            qc = .5*xc'*G*xc + c'*xc
            nlwc = length(Wci)
            println("i=$iter,\t q(x)= $qc, \t numel(Wci)=$nlwc")
        end

        # println(" Current xc = $xc")

        # update gradient of QP
        gc  = G*xc+c

        # update pc and lambda of inequality constraints
        pc,lami = solveKKT(xc,G,gc,Ae,be,Aic,bi[Wci])
        # pc,lami = solveSchur(xc,G,gc,Ae,be,Aic,bi[Wci],Ginv)

        # println("\n ##### iter $iter ##### \n pc $pc \t lami ", lami)

        if norm(pc)<tolP # if stepsize to x* is small
            # check Lagrange multipliers
            if all(lami.>=0) # we are done!
                break #return xc,iter,Wc
            else # remove constraint
                l_, idm = findmin(lami); # find inequality working set index corresponding to smallest value of multiplier
                Wci = deleteat!(Wci,idm) # remove one inequality constraint
                # println("removed one inequality constraint, new Wci $Wci")
                Aic  = Ai[Wci,:] # update active constraints
            end
        else
            # add one constraint to active set
            t2 = map(x -> min(0,x), (bi-Ai*xc)) # keep only negative entries that satisfy inequaity constraint, t2 has only entries <= 0, necessary to ensure alpha > 0
            t1 = (Ai*pc);
            t  = t2./t1;
            # println("bi-Ai*xc ", bi-Ai*xc)
            # println("\n t = t2./t1  $t \n t1 = Ai*pc $t1")
            # println("\n t2 = min(0,bi-Ai*xc) $t2")
            # println("t1 >= 0 \t", 1*(t1.>=0))
            t[find(t1 .>= 0)] = 1; # set entries in t equal 1 if t1 entries >= 0, necessary to ensure alpha > 0
            t[Wci]    = 1;
            # println("t[Wci]=1 $t")
            alpha = min(1.0,minimum(t)); # minimum for finding array min only, find step size
            if alpha<0
                error("alpha negative");
            end
            xc = xc + alpha*pc;
            # println("xc + alpha*pc $xc")
            if alpha<1 # add one blocking constraint
                idm = findin(t, alpha); # find indicies of entries in t == alpha
				# Wci = sort([Wci;idm[1]]); # sort to ensure index for inequality constraints are above equality constraints
				Wci = [Wci;idm[1]];
                Aic  = Ai[Wci,:] # found working set
            end
        end
        # check stopping
        if iter > maxIter
            break
        end
        iter = iter+1;
    end
    return xc, iter, Wci
end

"""
 function solveKKT(xc,G,gc,Ae,be,Ai,bi)

 Solves KKT system

       | G   Ae'  Ai' | |  p   |   | -gc |
 KKT = | Ae   0   0   | | -lame| = | -he |
       | Ai   0   0   | | -lami|   | -hi |

 gc = c+G*xc, he = Ae*xc-be, hi = Ai*xc-bi

 Input:
	 1. xc	  - current point, nx1
	 1. G 	  - matrix of weights for x, nxn
	 2. gc 	  - gradient of quadratic problem, nx1
	 3. Ai/Ae - matrix of inequality/equality constraints, mxn or pxn
	 4. bi/be - inequality/equality constraints to satisfy, mx1 or px1

Output:
	1. p 	- stepsize in direction of satisfying both inequality and equality constraints to update xc
	2. lami - Lagrange multipliers associated with inequaity constraints

"""
function solveKKT(xc,G,gc,Ae,be,Ai,bi)

    hi = Ai*xc-bi # h = 0, x0 = 0, b = 0
    he = Ae*xc-be

    KKT = [G Ae' Ai';
           Ae zeros(size(Ae,1), size(Ae,1)+size(Ai,1));
           Ai zeros(size(Ai,1), size(Ae,1)+size(Ai,1)) ]

	if rank(KKT) < size(KKT,1)
	   KKT = [G+1e-6*eye(size(G,1)) Ae' Ai';
	           Ae zeros(size(Ae,1), size(Ae,1)+size(Ai,1));
	           Ai zeros(size(Ai,1), size(Ae,1)+size(Ai,1)) ]
	end
    # println("solveKKT $KKT \n rank(KKT) ", rank(KKT))
    # println("\n ###### solveKKT ###### \n rank(KKT) $(rank(KKT)) \t size(KKT) ", size(KKT), " \t cond(KKT) ", cond(KKT), "\n xc $xc \n gc $gc \n G $G \n cond(G) ", cond(G)) #1e6 or larger

    rhs = [-gc; -he; -hi];

    try
        xsol = KKT\rhs;
        # println("inside try xsol $xsol")
        catch E
        println("ERROR!!!! \n rank(KKT) $(rank(KKT)) \t size(KKT) ", size(KKT), " \t cond(KKT) ", cond(KKT), "\n xc $xc \n gc $gc \n Ai $Ai \n bi $bi \n be $be \n Ae $Ae \n G $G \n rank(G) ", rank(G)) #1e6 or larger
        # JLD.save("asqp error case.jld","G", G, "xc", xc, "gc", gc, "Ai", Ai, "bi", bi, "Ae", Ae, "be", be)
        # MAT.matwrite("test.mat", Dict(
                #"G" => G,
                #"xc" => xc,
                #"gc" => gc,
                #"Ai" => Ai,
                #"bi" => bi,
                #"Ae" => Ae,
                #"be" => be,
                #"KKT" => KKT
                #))
        throw(E)
    end

    xsol = KKT\rhs
    p = xsol[1:length(gc)]
    lame = -xsol[length(gc)+(1:size(Ae,1))];
    lami = -xsol[length(gc)+size(Ae,1)+1:end];

    # println("\n p and lam $xsol \n Ae $Ae \n Ai ", Ai)

    return p,lami
end
#
#
# """
#  solveSchur(xc,G,gc,Ae,be,Ai,bi,Ginv=x->G\\x)
#
#  Solves KKT system
#
#        | G   Ae'  Ai' | |  p   |   | -gc |
#  KKT = | Ae   0   0   | | -lame| = | -he |
#        | Ai   0   0   | | -lami|   | -hi |
#
#  using Schur complement
#
#  he = Ae*xc-be, gc = c+G*xc, hi = Aic*xc-bic
#
# """
# function solveSchur(xc,G,gc,Ae,be,Ai,bi,Ginv=x->G\x)
#     # println("##### solve Schur #######")
#
#     A = [Ae;Ai]
#     b = [be;bi]
#     elen = length(be)
#     #println("A $A \t x $xc \t b $b \n G $G \t Wc $Wc")
#
#     h   = A*xc-b
#     if isempty(A)
#         lami = zeros(0)
#         p    = Ginv(-gc)
#     else
#         lam  = (A*(Ginv(A')))\(A*Ginv(gc)-h)  # note that lam has a minus sign in KKT system
# # println("lam = $lam")
# # println("A'*lam-gc=$(A'*lam-gc)")
#         p    = Ginv(A'*lam-gc)
#         lami = lam[elen+1:end]
#     end
#     return p, lami
# end

"""
function getLinearIndependentConstraints(A,Wc)

Find linearly independent rows of A (working set of constraints are independent of each other)

Input:
	1. A  - set of constraints in matrix form, matrix of integers
	2. Wc - current working set of constraints, array of integers

Output:
	1. Wc - updated working set of constraints, array of integers

"""

function getLinearIndependentConstraints(A,Wc)
    if length(Wc)==1 # only working constraint
        return Wc
    end
    # save('/tmp/getLinearIndependentConstraints.mat')

    # println("##### getLinearIndependentConstraints ###### \n A $A \n Wc $Wc")
    # make matrix smaller using its sparsity
    A  = A[Wc,:];
    id = find(0.< sum(abs.(A),1)); # sum along colums of A > 0
    # println("id $id \n A $A")
    A = A[:,id[:]]; # restrict to nonzero columns
    # println("A ", A)

    # check if rank is full
    if maximum(size(A)) <=3000
        if rank(full(A)) == length(Wc)
            #println("full rank A")
            return Wc
        end
    end

    debit = 1;
    credit = 2:size(A,1);

    # eliminate linear dependent constraints row by row
    Aac = A[debit,:];
    # println("Aac before credit ", Aac)

    while !(isempty(credit))
        # tic()
        At  = A[credit,:]';
        # println("At ", At, "\n Aac ", size(Aac)) #row vector always as columns
        xx = (Aac)\At; # solve system after adding new row (constraints)
        # println("xx ", xx)
        res = full(Aac*xx-At); # convert sparse to full matrix
        # println("res ", res)
        nres = sum(res.^2,1); # residual sum of squares along column
        mx,ii = findmax(nres); # find column with largest RSS
        # println("max is $mx")

        if mx>1e-3
            debit  = [debit;credit[ii]];
            # println("debit ", debit)
            credit = deleteat!(Vector(credit),ii) # remove row of constraints from inactive set
            # println("credit ", credit, "\n At[:,ii]' ", size(At[:,ii],2), "\ Aac ", Aac)
            Aac = [Aac'; At[:,ii]']; # update active set of constraints
            # println("credit ", credit, "\n Aac ", Aac)
        else
            break;
        end
        # toc()
    end

    Wc = Wc[debit]
    return Wc
end
