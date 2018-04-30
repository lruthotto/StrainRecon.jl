using StrainRecon
using Base.Test

 @testset "findw" begin

n    = 2
R     = StrainRecon.sdiag(5*ones(2))
M     = BitArray(eye(2))
Data  = ones(2)

paramGauss = getStrainReconParam(Data,n,R=R,pwr=2.0)
#paramLaplace = getStrainReconParam(Data,n,R=R,pwr=1.0)
#param = paramLaplace
for param=[paramGauss]
	@testset "findw pwr=$(param.pwr)" begin
		a     = findw(param,M)

		@test norm(a-[.5;.5])<1e-10

		param.d = ones(2)
		a    = findw(param,M)
		@test norm(a-[.5;.5])<1e-10

		param.d = [190;0]
		a    = findw(param,M)
		@test norm(a-[1;0])<1e-10

		param.d = [0;20]
		a    = findw(param,M)
		@test norm(a-[.5;.5])<1e-10

		M    = BitArray([1.0 0; 0 0])
		param.d = ones(2)
		a    = findw(param,M)
		@test norm(a-[1;0])<1e-10
	end
end

    n    = 2
    R     = StrainRecon.sdiag(5*ones(2))
    M     = BitArray([1 0; 1 0])
    Data  = ones(2)
    paramLaplace = getStrainReconParam(Data,n,R=R,pwr=1.0)

for param=[paramLaplace]
        @testset "findw pwr=$(param.pwr)" begin
            a     = findw(param,M)
            # println("a = $a")
            @test norm(a-[1;0])<1e-10

            param.d = zeros(2)
            a    = findw(param,M)
            # println("a = $a")
            @test norm(a-[0.5;0.5])<1e-10

            M    = BitArray([1.0 0; 0 0])
            param.d = ones(2)
            a    = findw(param,M)
            # println("a = $a")
            @test norm(a-[1;0])<1e-10

            m = 24
            for n = 2:5
                param.R = StrainRecon.sdiag(5*ones(m))
                param.n = n
                # srand(s)
                M = rand([0,1], m, n)
                atrue = randomw(n)
                param.d = M*atrue
                M = BitArray(M)
                a    = findw(param,M)
                # println("a = $a , atrue = $atrue")
                @test norm(a-atrue)<1e-10
            end

        end
    end
end
