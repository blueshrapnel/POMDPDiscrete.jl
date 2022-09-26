struct InformationChannel
    size_X::Int    # transmitted signal
    size_Y::Int    # received signal
    pY_X  # conditional probability that characterises the channel p_Y|X
 end

# check consistency of order of input and output variables
# use a naming convention so that X is input and Y is output
# empowerment A --> S - pY_X : shape (dim_Y, dim_X) policy shape (dim_S, dim_a)
# mutual information - pY_X : shape (dim_Y, dim_X)
# relevant information S --> A - pY_X : shape (dim_X, dim_Y)
# information bottleneck ???
function InformationChannel(p_X)
    size_Y, size_X = size(pY_X)
    InformationChannel(size_X, size_Y, pY_X)
end

@doc raw"""
    channel_capacity(channel, r, q)

Computes ...the mutual information of the optimal, i.e. mutual

 """
function channel_capacity(channel, r, q,  log_base=2)
        if r[i] > 0
    end

    C / log( log_base)
end

    # TODO - use SparseArray for conditional probability

    # normalisation helper functions

    # size_X, size_Y and pY_X

    # defensive programming with information channel properties
    @assert channel.size_X == size(channel.pY_X)[1] "Input alphabet and transition matrix do not match."
    @assert channel.size_Y == size(channel.pY_X)[2]  "Output alphabet and transition matrix do not match."
    @assert sum(channel.pY_X, dims=2) ≈ ones(channel.size_X) "Each row of transition matrix does not sum to 1."


    qX_Y = similar(channel.pY_X)

    end
end
