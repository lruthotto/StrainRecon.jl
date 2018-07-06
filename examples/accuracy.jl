#==============================================================================
This script generates random measurements and computes MAP estimates using the
block coordinate descent (BCD) method and the convex mixed-integer quadratic
programming (MIQP) approach. The latter relies on Gurobi software (see also
solveConvexMAP.jl). The reconstruction errors are computed and their
distributions are plotted.

N is the number of samples for a given set of parameters m, n, and p. For large N,
running this script may take some time.

Parallelization can be enabled, e.g., by starting julia with "julia -p 16".
Visualization uses the PyPlot package.

This is the set of experiments presented with N=10000 in Section 4.3 of the article:

L. Mustonen, X. Gao, A. Santana, R. Mitchell, Y. Vigfusson, and L. Ruthotto,
A Bayesian framework for molecular strain identification from mixed diagnostic samples
https://arxiv.org/abs/1803.02916
==============================================================================#

@everywhere using StrainRecon
using PyPlot
srand(1) # for reproducibility

m_range = [10] # number of measurement locations
n_range = [3,4] # number of strains
p_range = [2,4] # number of categories in each location

noiseVar_range = [1e-6, 1e-4] # variance of the added Gaussian noise in the data

N = 100 # number of samples for each triplet (m,n,p)

nT = 20 # number of trials in the block coordinate descent method

normp = 1.0 # exponent in the distance function

# ground truths and synthetic measurements
Mtrue  = Array{Array}(length(m_range), length(n_range), length(p_range))
wtrue  = Array{Array}(length(m_range), length(n_range), length(p_range))
dnoise = Array{Array}(length(m_range), length(n_range), length(p_range), length(noiseVar_range))

# reconstructions
MrecoBCD  = Array{Array}(length(m_range), length(n_range), length(p_range), length(noiseVar_range))
wrecoBCD  = Array{Array}(length(m_range), length(n_range), length(p_range), length(noiseVar_range))
MrecoMIQP  = Array{Array}(length(m_range), length(n_range), length(p_range), length(noiseVar_range))
wrecoMIQP  = Array{Array}(length(m_range), length(n_range), length(p_range), length(noiseVar_range))

# reconstruction errors and expected distances
recoerrBCD = Array{Array}(length(m_range), length(n_range), length(p_range), length(noiseVar_range))
recoerrMIQP = Array{Array}(length(m_range), length(n_range), length(p_range), length(noiseVar_range))
aveDist = zeros(length(m_range), length(n_range), length(p_range))

for m_idx = 1:length(m_range)
    m = m_range[m_idx]

    for n_idx = 1:length(n_range)
        n = n_range[n_idx]

        for p_idx = 1:length(p_range)
            p = p_range[p_idx]

            # generate data for the current triplet (m,n,p)
            Mtrue_mnp = randomM(m,n,p=p,N=N)
            wtrue_mnp = randomw(n,N)
            dtrue = [Mtrue_mnp[:,:,idx]*wtrue_mnp[:,idx] for idx in 1:N]
            dtrue = hcat(dtrue...) # convert to (n,N)
            Mtrue[m_idx,n_idx,p_idx] = Mtrue_mnp
            wtrue[m_idx,n_idx,p_idx] = wtrue_mnp

            # compute expected distance
            Mrand = randomM(m,n,p=p,N=N)
            wrand = randomw(n,N)
            aveDist[m_idx,n_idx,p_idx] = mean(srDistance(Mtrue_mnp,wtrue_mnp,Mrand,wrand,p=p,pn=normp))

            for noiseVar_idx = 1:length(noiseVar_range)
                noiseVar = noiseVar_range[noiseVar_idx]
                R = 1/sqrt(noiseVar)

                # create noisy data
                dnoise_mnpv = dtrue + sqrt(noiseVar)*randn(size(dtrue))
                dnoise[m_idx,n_idx,p_idx,noiseVar_idx] = dnoise_mnpv

                # reconstruct using block coordinate descent
                Mreco_mnpv = SharedArray{Int8}(m*(p-1),n,N)
                wreco_mnpv = SharedArray{Float64}(n,N)
                @sync @parallel for idx = 1:N
                    param = getStrainReconParam(dnoise_mnpv[:,idx],n,p=p,R=R)
                    Mmap, wmap = solveMAP(param,maxTry=nT)
                    Mreco_mnpv[:,:,idx] = Mmap
                    wreco_mnpv[:,idx] = wmap
                end

                # store the reconstructions and reconstruction errors for BCD
                MrecoBCD[m_idx,n_idx,p_idx,noiseVar_idx] = convert(BitArray{3},Mreco_mnpv)
                wrecoBCD[m_idx,n_idx,p_idx,noiseVar_idx] = convert(Array{Float64,2},wreco_mnpv)
                recoerrBCD[m_idx,n_idx,p_idx,noiseVar_idx] =
                    srDistance(BitArray(Mtrue[m_idx,n_idx,p_idx]),wtrue[m_idx,n_idx,p_idx],convert(BitArray{3},Mreco_mnpv),convert(Array{Float64,2},wreco_mnpv),p=p,pn=normp)

                # reconstruct using MIQP (Gurobi does not need parallel loop)
                Mreco_mnpv = BitArray(m*(p-1),n,N)
                wreco_mnpv = zeros(n,N)
                for idx = 1:N
                    param = getStrainReconParam(dnoise_mnpv[:,idx],n,p=p,R=R)
                    Mmap, wmap = solveConvexMAP(param)
                    Mreco_mnpv[:,:,idx] = Mmap
                    wreco_mnpv[:,idx] = wmap
                end

                # store the reconstructions and reconstruction errors for MIQP
                MrecoMIQP[m_idx,n_idx,p_idx,noiseVar_idx] = convert(BitArray{3},Mreco_mnpv)
                wrecoMIQP[m_idx,n_idx,p_idx,noiseVar_idx] = wreco_mnpv
                recoerrMIQP[m_idx,n_idx,p_idx,noiseVar_idx] = srDistance(BitArray(Mtrue[m_idx,n_idx,p_idx]),wtrue[m_idx,n_idx,p_idx],convert(BitArray{3},Mreco_mnpv),wreco_mnpv,p=p,pn=normp)
            end

            println("m", m, " n", n, " p", p) # show progress

        end
    end
end
# at this point you may want to save all the computed variables

# visualize errors (load variables first if necessary)
m_idx = 1
m = m_range[m_idx]

for n_idx = 1:2
    n = n_range[n_idx]

    for p_idx = 1:2
        p = p_range[p_idx]

        subplot(2,2,n_idx+2*floor(p_idx/2))

        # plot errors in ascending order
        plot(sort(recoerrBCD[m_idx,n_idx,p_idx,2]))
        plot(sort(recoerrMIQP[m_idx,n_idx,p_idx,2]))
        plot(sort(recoerrBCD[m_idx,n_idx,p_idx,1]))
        plot(sort(recoerrMIQP[m_idx,n_idx,p_idx,1]))

        plot(aveDist[m_idx,n_idx,p_idx]*ones(N))

        xlabel("sample")
        ylabel("error")
        title("m=$m, n=$n, p=$p")
        if n_idx == 1 && p_idx == 1
            legend(("BCD $(sqrt(noiseVar_range[2]))","MIQP $(sqrt(noiseVar_range[2]))","BCD $(sqrt(noiseVar_range[1]))","MIQP $(sqrt(noiseVar_range[1]))","ave. dist."))
        end
    end
end
