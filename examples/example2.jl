#==============================================================================
For a given measurement vector, this example computes a MAP estimate, all global
minimizers of the log-likelihood, posterior means, and posterior standard
deviations.

In addition, we visualize the posterior density and the entropy of the
conditional posterior of M as functions of w.

Additional packages may be needed to install for the visualization.

This is the second example presented in Section 4.2 of the article:

L. Mustonen, X. Gao, A. Santana, R. Mitchell, Y. Vigfusson, and L. Ruthotto,
A Bayesian framework for molecular strain identification from mixed diagnostic samples
https://arxiv.org/abs/1803.02916
==============================================================================#

using StrainRecon
using PyPlot
using jInv.Utils
using jInv.Mesh
using jInvVis
srand(0) # initialize the random number generator for reproducibility

# define the measurement vector
d = [.1,.3,.5,.6]

# dimensions of the problem (the number of classes is implicitly p=2)
m = length(d)
n = 3

R = 100 # inverse square root of the diagonal entry in the matrix Gamma
param = getStrainReconParam(d,n,R=R)

# generate all possible binary matrices of size (m,n)
Mbits = [[bit=='1' for bit in bin(k,m*n)] for k in 0:2^(m*n)-1]
Mall = true*reshape(hcat(Mbits...),(m,n,2^(m*n)))
N = size(Mall,3)

# compute minimizing frequency vectors and the corresponding misfits for each
# binary matrix
wmins = zeros(n,N)
misfits = zeros(N)
for idx = 1:N
    wmins[:,idx] = findw(param,Mall[:,:,idx])
    misfits[idx] = misfit(param,Mall[:,:,idx],wmins[:,idx])[1]
end

# sort misfits in ascending order
srt = sortperm(misfits)

# print all pairs (M,w) that correspond to the minimum misfit
for idx = 1:N
    if abs(misfits[srt[idx]] - misfits[srt[1]]) > 1e-8
        break # not a global minimum anymore
    end
    Base.showarray(STDOUT,1*Mall[:,:,srt[idx]],false)
    println()
    println(wmins[:,srt[idx]])
end
println()

# compute posterior means and variances
Mcm,wcm,dummy,Mvar,wvar = wrapperIntegratePosterior(param)

println("Posterior means:")
Base.showarray(STDOUT,Mcm,false)
println()
println(wcm)
println()

println("Posterior standard deviations:")
Base.showarray(STDOUT,sqrt.(Mvar),false)
println()
println(sqrt.(wvar))
println()

# visualize posterior (reduced misfit) and entropy
Ma = getRegularMesh([1/3 1 0 1/2 ],3*[80 80])
as = getCellCenteredGrid(Ma)
redmisfits = zeros(Ma.nc)
redmisfits[:] = NaN # white background
entropy = zeros(Ma.nc)
entropy[:] = NaN

for k=1:size(as,1)
    at = vec([ vec(as[k,:]); 1-sum(as[k,:])])
    if all(at.>=0) && sum(at)<=1 && all(diff(at).<=0)
        redmisfits[k] = 2*reducedMisfit(param,at)[1] # drop the coefficient 1/2
        entropy[k] = entropyM(param,at)
    end
end

subplot(1,2,1)
viewImage2D(redmisfits,Ma)
xlabel("w1")
ylabel("w2")
axis("image")
title("Reduced misfit")
colorbar()
set_cmap("Spectral")
clim(0,240)

subplot(1,2,2)
viewImage2D(entropy,Ma)
xlabel("w1")
ylabel("w2")
axis("image")
title("Entropy")
colorbar()
set_cmap("Spectral")
clim(0,3)

# fix w and compute matrix minimizers
w0 = [0.5; 0.3; 0.2]
misfits0 = zeros(N)
for idx = 1:N
    misfits0[idx] = misfit(param,Mall[:,:,idx],w0)[1]
end

# sort misfits in ascending order
srt0 = sortperm(misfits0)

# print all matrices that correspond to the minimum misfit for the given w
for idx = 1:N
    if abs(misfits0[srt0[idx]] - misfits0[srt0[1]]) > 1e-8
        break # not a minimizing matrix anymore
    end
    Base.showarray(STDOUT,1*Mall[:,:,srt0[idx]],false)
    println()
end
