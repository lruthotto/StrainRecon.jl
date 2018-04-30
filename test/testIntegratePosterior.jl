@testset "integratePosterior" begin

function dotest(param,Mmeanref,wmeanref,wstdref)
	d = param.d
	m = param.m
	n = param.n
	p = param.p

	rowfuns = [(r,w)->w;repmat([(r,w)->zeros(n,1)],m-1)]
	wmean = integratePosterior(param,rowfuns)
	@test norm(wmean-wmeanref)/norm(wmeanref) < 5e-3

	if !isempty(wstdref)
		rowfuns = [(r,w)->(w-wmean).^2;repmat([(r,w)->zeros(n,1)],m-1)]
		wvar = integratePosterior(param,rowfuns)
		wstd = sqrt.(wvar)
		@test norm(wstd - wstdref)/norm(wstdref) < 1
	end

	rowfuns = Array{Function,1}()
	for i = 1:m
		z1 = zeros((i-1)*(p-1),n)
		z2 = zeros((m-i)*(p-1),n)
		push!(rowfuns, (r,w)->[z1;r;z2])
	end
	Mmean = integratePosterior(param,rowfuns)
	@test norm(Mmean-Mmeanref)/norm(Mmeanref) < 1e-6
end

# Common properties
gamma = 1e-6
lambda = .0


# Case 1
data = [.4, .6, 1]
p = 2
Mmeanref = [0 1; 1 0; 1 1]
wmeanref = [.6, .4]
wstdref = [0.000710, 0.000710]

R = gamma^(-.5) * StrainRecon.sdiag(ones(length(data)))
param = getStrainReconParam(data,length(wmeanref),R=R,p=p,lambda=lambda)
dotest(param,Mmeanref,wmeanref,wstdref)


# Case 2
data = [.5, .5, 1]
p = 2
Mmeanref = [.5 .5; .5 .5; 1 1]
wmeanref = ([.5, .5])
wstdref = ([0.000424, 0.000424])

R = gamma^(-.5) * StrainRecon.sdiag(ones(length(data)))
param = getStrainReconParam(data,length(wmeanref),R=R,p=p,lambda=lambda)
dotest(param,Mmeanref,wmeanref,wstdref)


# Case 3
data = [.4, 0, 0, .6, 1, 0]
p = 3
Mmeanref = [0 1; 0 0; 0 0; 1 0; 1 1; 0 0]
wmeanref = [.6, .4]
wstdref = []

R = gamma^(-.5) * StrainRecon.sdiag(ones(length(data)))
param = getStrainReconParam(data,length(wmeanref),R=R,p=p,lambda=lambda)
dotest(param,Mmeanref,wmeanref,wstdref)

end
