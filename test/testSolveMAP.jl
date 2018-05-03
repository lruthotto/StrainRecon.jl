
@testset "solveMAP" begin

M  = [1 1 1; 1 1 0; 1 0 1; 1 0 0; 0 1 1; 0 1 0; 0 0 1;]
(m,n) = size(M)
R     = StrainRecon.sdiag(ones(m)*100);
atrue = [.6; 0.3; 0.1]
Data  = M*atrue

param = getStrainReconParam(Data,n,R=R)

Mmap,amap = solveMAP(param,maxTry=100,maxIterQP=10)
@test norm(amap - atrue)/norm(atrue) < 1e-8
@test norm(Mmap*amap-Data) < 1e-7

Msqp,Asqp = solveMAP(param,storeInterm=true,maxIterQP=20)
# Asqp is n-by-history of each BCD iteration-by-MaxTry
# Msqp is m-by-n-by-history of each BCD iteration-by-MaxTry

Jc = zeros(size(Asqp,3)) # Misfit for MaxTry number of iterations

for k=1:size(Asqp,3)
    # for each Asqp iteration, find column where norm(w_old - w_new) = 0, use column number - 1 to find frame for Msqp
    iter = size(Asqp,2)
    for i = 1:size(Asqp,2)
        # println(i)
        if sum(Asqp[:,i,k]) == 0
            iter = i-1
            # println("found column of all zeros ", i-1)
            break
        end
    end
    @test all(Asqp[:,iter,k].>0)
	Jc[k] = misfit(param,Msqp[:,:,iter-1,k],Asqp[:,iter,k])[1]
end
@test all(diff(Jc).>=0)
@test norm(sum(Asqp[:,1],1)-1,Inf) < 1e-7 # Asqp is n-by-history of iterations
# @test all(Asqp[:,:].>0)
end
