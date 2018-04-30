export intmat2binmat

"""
function intmat2binmat(M;p=maximum(M))

Converts a general integer matrix {1,2,...,p}^{m x n} to a binary matrix of
size (m*(p-1),n).

Ones are converted to false vectors of length p-1. Two is converted to
[true;false;...;false] and p is converted to [false;...;false;true].

Input:
    M - Integer matrix of size (m,n) with integers 1,...,p

Keyword argument:
    p - number of categories (if larger than the largest number already in M)

Output:
    Mbin - BitArray of size (m*(p-1),n)
"""

function intmat2binmat(M::Array{Int,2};p::Int=maximum(M))
    if minimum(M)<=0
        error("the categories in the integer matrix must be positive integers")
    end

    m = size(M,1)
    n = size(M,2)
    p = maximum((p,maximum(M)))

    # initialize output
    Mbin = falses(m*(p-1),n)

    for k = 1:m
        fullblock = falses(p,n)

        # values in the k'th row of the original matrix are now row indices for
        # the next block in the binary matrix
        fullblock[sub2ind(size(fullblock),M[k,:],1:n)] = true

        # discard the first row because it is redundant
        block = fullblock[2:p,:]

        # assign the new block to the output matrix
        Mbin[(k-1)*(p-1)+1 : k*(p-1), :] = block
    end

    return Mbin
end
