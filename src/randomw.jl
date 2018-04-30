export randomw
"""
function randomw(n,nsamp=1)

Samples from a uniform distribution in the set

{w in R^n |  sum(w)=1, diff(w).<=0, w.>=0}

Input:

    1. n        - number of strains

Optional argument:

    2. nsamp    - number of samples (default=1)

Output:

    1. wsamples - a matrix of size (n,nsamp) or a vector of length n if nsamp==1
"""
function randomw(n::Int64,nsamp::Int64=1)

    if n == 1
        return nsamp == 1 ? ones(1) : ones(1,nsamp)
    end

    wsamples = zeros(n, nsamp)

    for k = 1:nsamp
        w = [0; sort!(rand(n-1)); 1]
        w = sort(diff(w), rev=true)
        wsamples[:, k] = w;
    end

    if nsamp==1
        return vec(wsamples)
    else
        return wsamples
    end

end
