using StrainRecon
using Base.Test

@testset "findM" begin
    n = 3
    m = 5
    nsamp = 5
    w = randomw(n,nsamp)
    Mtrue = randomM(m,n)
    for i = 1:nsamp
        d = Mtrue*w[:,i]
        param = getStrainReconParam(d,n)
        M_nsol,pm = findM(param, w[:,i], nsol = 2^n, probM = true)
        for j = 1:2^n
            M = M_nsol[:,:,j]*1
            #@test M*w[:,i] 
        end
        
        @test all(diff(pm,2) .<= 0) # check that subsequent rows of M in nth + 1 matrix are not better than previous matrix
        
        @test norm(sum(pm,2) .- 1) < 1e-8 # sum of p(M) across matrices at the same row = 1
        
    end
end
        