#==============================================================================
This script reconstructs the binary matrix and frequency vector from experimental
measurement data stored in MAT file realdata.mat.

In addition, the dependence between the frequency vector and the reconstruction
error is examined with synthetic data which is generated with the given binary
matrix, different frequency vectors, and additive Gaussian noise.

Parallelization can be enabled, e.g., by starting julia with "julia -p 16".
Additional packages may be needed to install for the visualization.

This is the set of experiments presented (with nT=200 and higher resolution) in
Section 4.4 of the article:

L. Mustonen, X. Gao, A. Santana, R. Mitchell, Y. Vigfusson, and L. Ruthotto,
A Bayesian framework for molecular strain identification from mixed diagnostic samples
https://arxiv.org/abs/1803.02916
==============================================================================#

@everywhere using StrainRecon
using PyPlot
using jInv.Utils
using jInv.Mesh
using jInvVis
using MAT
srand(2) # for reproducibility

# load data from MAT file
data = matread("realdata.mat")
M_true = data["M_true"]
w_true = vec(data["w_true"])
d_meas = vec(data["d_meas"])

m = length(d_meas) # number of measurements
n = 3 # number of strains in the reconstruction
R = 10 # variance of the measurement noise is assumed to be 0.01
param = getStrainReconParam(d_meas,n,R=R)
nT = 200 # number of trials in BCD

# compute sample standard deviation of the error
errstd = sqrt(var(M_true*w_true-d_meas))
println("Sample standard deviation of the measurement error is $errstd")

# compute MAP estimates
MrecoBCD,wrecoBCD = solveMAP(param,maxTry=nT)
MrecoMIQP,wrecoMIQP = solveConvexMAP(param)

# compute posterior means and variances
Mcm,wcm,dummy,Mvar,wvar = wrapperIntegratePosterior(param)

# show results
imcolor = "gray"
imlimits = [.5, m+.5, n+.5, .5]

subplot(5,1,1)
imshow(M_true',cmap=imcolor,extent=imlimits,vmin=0,vmax=1)
title("true M")
println("true w:")
println(w_true)
println()

subplot(5,1,2)
imshow(MrecoBCD',cmap=imcolor,extent=imlimits,vmin=0,vmax=1)
title("M from BCD")
println("w from BCD:")
println(wrecoBCD)
println()

subplot(5,1,3)
imshow(MrecoMIQP',cmap=imcolor,extent=imlimits,vmin=0,vmax=1)
title("M from MIQP")
println("w from MIQP:")
println(wrecoMIQP)
println()

subplot(5,1,4)
imshow(Mcm',cmap=imcolor,extent=imlimits,vmin=0,vmax=1)
title("posterior mean of M")
println("posterior mean of w:")
println(wcm)
println()

subplot(5,1,5)
imshow(sqrt.(Mvar)',cmap=imcolor,extent=imlimits,vmin=0,vmax=1)
title("posterior standard deviation of M")
println("posterior standard deviation of w:")
println(sqrt.(wvar))
println()

# keep M fixed, vary w and reconstruct from synthetic, noisy measurements
noiseVar_range = [1e-2, 1e-4]
nT = 20 # decrease the number of trials for efficiency
resolution = 64
Ma = getRegularMesh([1/3 1 0 1/2 ],[resolution resolution])
as = getCellCenteredGrid(Ma)
recoerrs = SharedArray{Float64}(Ma.nc,2)
recoerrs[:] = NaN # white background

for noiseVar_idx = 1:2
    @sync @parallel for k=1:size(as,1)
        at = vec([ vec(as[k,:]); 1-sum(as[k,:])])
        if all(at.>=0) && sum(at)<=1 && all(diff(at).<=0)
            d = M_true * at + sqrt(noiseVar_range[noiseVar_idx])*randn(m)
            param = getStrainReconParam(d,n)
            Mreco,wreco = solveMAP(param,maxTry=nT)
            recoerrs[k,noiseVar_idx] = srDistance(BitArray(M_true),at,BitArray(Mreco),wreco,pn=1.0)
        end
    end
end

figure()
subplot(1,2,1)
viewImage2D(recoerrs[:,1],Ma)
xlabel("w1")
ylabel("w2")
axis("image")
title("reconstruction error, noise $(sqrt(noiseVar_range[1]))")
colorbar()
set_cmap("Spectral")
clim(0,17)

subplot(1,2,2)
viewImage2D(recoerrs[:,2],Ma)
xlabel("w1")
ylabel("w2")
axis("image")
title("reconstruction error, noise $(sqrt(noiseVar_range[2]))")
colorbar()
set_cmap("Spectral")
clim(0,17)
