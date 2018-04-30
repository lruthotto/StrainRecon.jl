using StrainRecon
using Base.Test

@testset "BCD" begin
M  = [1 1 1; 1 1 0; 1 0 1; 1 0 0; 0 1 1; 0 1 0; 0 0 1;]
(m,n) = size(M)
R     = StrainRecon.sdiag(ones(m)*100);
atrue = [.6; 0.3; 0.1]
Data  = M*atrue

paramGauss = getStrainReconParam(Data,n,R=R,pwr=2.0)
paramLaplace = getStrainReconParam(Data,n,R=R,pwr=1.0)

for param=[paramGauss,paramLaplace]
    @testset "testBCD pwr=$(param.pwr)" begin
        maxTry = 10
        A0   = zeros(n,maxTry)
        Abcd = zeros(n,maxTry)
        Mbcd =  falses(m,n,maxTry)
        Jbcd = zeros(maxTry)
        mbcd = falses(m,n)
        for k=1:maxTry
            a0 = randomw(n)
            abcd,mbcd,his = bcd(param, a0,out=-1)
            Abcd[:,k] = abcd
        	Mbcd[:,:,k] = mbcd
            A0[:,k] = a0
            Jbcd[k] = misfit(param,mbcd,abcd)[1]
        end
        # find global minimizer
        kbcd = indmin(Jbcd)
        abcd = Abcd[:,kbcd]
        Mbcd = Mbcd[:,:,kbcd]

        @test norm(sum(Abcd,1)-1,Inf) < 1e-8
        @test norm(abcd - atrue)/norm(atrue) < 1e-8
        @test all(Abcd.>=0)
        @test all(Mbcd.==M)
        @test norm(Mbcd*abcd-Data)/norm(Data) < 1e-8
    end
end
end
