#==============================================================================
This example generates two noiseless measurement vectors corresponding to one
binary matrix and two different frequency vectors.

For each measurement vector, we compute MAP estimates, all global minimizers of
the negative log-likelihood, posterior means, and posterior standard deviations.

This is the first set of examples presented in Section 4.2 of the article:

L. Mustonen, X. Gao, A. Santana, R. Mitchell, Y. Vigfusson, and L. Ruthotto,
A Bayesian framework for molecular strain identification from mixed diagnostic samples
https://arxiv.org/abs/1803.02916
==============================================================================#

using StrainRecon
srand(0) # initialize the random number generator for reproducibility

# binary matrix and frequency vectors (ground truths)
M = [0 1; 1 0; 1 1]
w1 = [0.6; 0.4]
w2 = [0.5; 0.5]

# measurement data (without noise)
d1 = M*w1
d2 = M*w2

# dimensions of the problem (the number of classes is implicitly p=2)
m = size(M,1)
n = size(M,2)

R = 100 # inverse square root of the diagonal entry in the matrix Gamma
param1 = getStrainReconParam(d1,n,R=R)
param2 = getStrainReconParam(d2,n,R=R)

# verify that the block coordinate descent solver finds a global minimum
Mreco1,wreco1 = solveMAP(param1)
Mreco2,wreco2 = solveMAP(param2)
misfit1 = misfit(param1,Mreco1,wreco1)[1]
misfit2 = misfit(param2,Mreco2,wreco2)[1]
println("Misfit for the first MAP estimate is $misfit1")
println("Misfit for the second MAP estimate is $misfit2")
println()

# generate all possible binary matrices of size (m,n)
Mbits = [[bit=='1' for bit in bin(k,m*n)] for k in 0:2^(m*n)-1]
Mall = true*reshape(hcat(Mbits...),(m,n,2^(m*n)))
N = size(Mall,3)

# compute minimizing frequency vectors and the corresponding misfits for
# each binary matrix
wmins1 = zeros(n,N)
misfits1 = zeros(N)
wmins2 = zeros(n,N)
misfits2 = zeros(N)
for idx = 1:N
    wmins1[:,idx] = findw(param1,Mall[:,:,idx])
    misfits1[idx] = misfit(param1,Mall[:,:,idx],wmins1[:,idx])[1]

    wmins2[:,idx] = findw(param2,Mall[:,:,idx])
    misfits2[idx] = misfit(param2,Mall[:,:,idx],wmins2[:,idx])[1]
end

# sort misfits in ascending order
srt1 = sortperm(misfits1)
srt2 = sortperm(misfits2)

# print all pairs (M,w) that correspond to the minimum misfit
println("Case 1 minimizers:")
for idx = 1:N
    if abs(misfits1[srt1[idx]] - misfits1[srt1[1]]) > 1e-8
        break # not a global minimum anymore
    end
    Base.showarray(STDOUT,1*Mall[:,:,srt1[idx]],false)
    println()
    println(wmins1[:,srt1[idx]])
end
println()

println("Case 2 minimizers:")
for idx = 1:N
    if abs(misfits2[srt2[idx]] - misfits2[srt2[1]]) > 1e-8
        break # not a global minimum anymore
    end
    Base.showarray(STDOUT,1*Mall[:,:,srt2[idx]],false)
    println()
    println(wmins2[:,srt2[idx]])
end

# compute posterior means and variances
Mcm1,wcm1,dummy1,Mvar1,wvar1 = wrapperIntegratePosterior(param1)
Mcm2,wcm2,dummy2,Mvar2,wvar2 = wrapperIntegratePosterior(param2)

println("Case 1 posterior means:")
Base.showarray(STDOUT,Mcm1,false)
println()
println(wcm1)
println()

println("Case 1 posterior standard deviations:")
Base.showarray(STDOUT,sqrt.(Mvar1),false)
println()
println(sqrt.(wvar1))
println()

println("Case 2 posterior means:")
Base.showarray(STDOUT,Mcm2,false)
println()
println(wcm2)
println()

println("Case 2 posterior standard deviations:")
Base.showarray(STDOUT,sqrt.(Mvar2),false)
println()
println(sqrt.(wvar2))
println()
