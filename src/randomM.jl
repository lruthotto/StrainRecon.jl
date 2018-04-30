export randomM

"""
function randomM(m,n;p=2,N=1)

Generate samples for M from uniform distribution with non-repeated columns.

Input:
    m - number of blocks
    n - number of columns
    p - number of categories in each block (default=2)
    N - number of samples (default=1)

Output:
    M - BitArray of size (m*(p-1),n,N), or (m*(p-1),n) if N==1
"""
function randomM(m::Int64,n::Int64;p::Int64=2,N::Int64=1)
    if log(n) > m*log(p)
        error("unique columns in M not possible if n > p^m")
    end

    # store all feasible blocks of M in a matrix B of size ((p-1)*p^n, n)
    B = getB(n,p)

    M = falses(m*(p-1),n,N)
    for idx = 1:N
        # rejection sampling: check uniqueness of columns
        firstFlag = true # (this should be a do-while loop)
        while firstFlag || size(unique(M[:,:,idx],2),2) < n
            firstFlag = false
            # randomly choose m blocks from B
            blockIdB = rand(1:p^n, m)
            for blockIdM = 1:m
                subsB = (blockIdB[blockIdM]-1)*(p-1)+1:blockIdB[blockIdM]*(p-1)
                subsM = (blockIdM-1)*(p-1)+1:blockIdM*(p-1)
                M[subsM,:,idx] = B[subsB,:]
            end
        end
    end

    # output
    if N==1
        M = squeeze(M,3)
    end
    return M
end
