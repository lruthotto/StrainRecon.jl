export srDistance

"""
function srDistance

Unnormalized distance (reconstruction error) between two pairs (M,w):

min_P ||M1*diagm(w1) - M2*diagm(w2)*P ||_p

where P is a permutation matrix and || ||_p denotes the entrywise matrix pn-norm
(i.e., pn=2 corresponds to the Frobenius norm).

Before computing the error, the first dimensions of the binary matrices are
augmented from m*(p-1) to m*p so that the column sums of each block become 1.
This can be disabled by setting p=1.

If n has different values among the inputs, then the largest n is chosen and
the rest are augmented with zeros to make the dimensions compatible.

Input:
    M1 - binary array of size (m*(p-1),n) or (m*(p-1),n,N)
    w1 - frequency vector of length n or an array of size (n,N)
    M2 - as M1
    w2 - as w1

Keyword arguments:
    p - number of categories (default=2)
    pn - exponent in the matrix norm (default=2.0)

Output:
    distance(s) as a scalar or an N-vector
"""

function srDistance(M1::BitArray{2},w1::Vector,M2::BitArray{2},w2::Vector;p::Int64=2,pn::Float64=2.0)
    n = maximum((size(M1,2),length(w1),size(M2,2),length(w2)))
    M1 = [M1 falses(size(M1,1),n-size(M1,2))]
    w1 = [w1; zeros(n-length(w1))]
    M2 = [M2 falses(size(M2,1),n-size(M2,2))]
    w2 = [w2; zeros(n-length(w2))]

    perms = permutations(1:n)

    if p>1
        M1 = fullCategoricalMatrix(M1,p)
        M2 = fullCategoricalMatrix(M2,p)
    end

    distance = Inf
    for perm in perms
        M2perm = M2[:,perm]
        w2perm = w2[perm]

        mat = M1*diagm(w1) - M2perm*diagm(w2perm)
        dist = (sum(abs.(mat[:]).^pn))^(1/pn)
        if dist < distance
            distance = dist
        end
    end

    return distance
end

function srDistance(M1::BitArray{3},w1::Array{Float64,2},M2::BitArray{3},w2::Array{Float64,2};p::Int64=2,pn::Float64=2.0)
    N = size(M1,3)
    distances = SharedArray{Float64}(N)

    @sync @parallel for idx = 1:N
        distances[idx] = srDistance(M1[:,:,idx],w1[:,idx],M2[:,:,idx],w2[:,idx],p=p,pn=pn)
    end

    return distances
end

function fullCategoricalMatrix(M::BitArray{2},p::Int64)
    n = size(M,2)

    M1 = reshape(M,(p-1,:))
    blockcolsums = sum(M1,1)
    Mfull = [1-blockcolsums; M1]

    return reshape(Mfull,(:,n))
end
