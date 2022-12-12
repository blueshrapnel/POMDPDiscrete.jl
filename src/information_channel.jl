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
    # this is simply calculating the mutual information for the optimal channel
    C = 0
    for i in 1:channel.size_X
        if r[i] > 0
            # if we have a deterministic channel, then Capacity is infinite
            # to circumvent and Inf value, we perturb zero values
            # also automatically handles 0 log 0 conventionally
            C += sum(r[i] .* channel.pY_X[i, :] .* log.(q[i,:]./r[i] .+ 1e-16))
        end
    end
    C / log( log_base)
end

function blahut_arimoto_channel_capacity(channel::InformationChannel, max_iter=50, ε=1e-4, log_base=2)
    # description of algorithm in Yeung "Information theory and network coding
    # normalisation helper functions
    norm_columns(A) = A./sum(A, dims=1)
    norm_rows(A) = A./sum(A, dims=2)

    # anticpate that the channel struct will contain fields:
    # size_X, size_Y and pY_X

    # defensive programming with information channel properties
    @assert channel.size_X == size(channel.pY_X)[1] "Input alphabet and transition matrix do not match."
    @assert channel.size_Y == size(channel.pY_X)[2]  "Output alphabet and transition matrix do not match."
    @assert sum(channel.pY_X, dims=2) ≈ ones(channel.size_X) "Each row of transition matrix does not sum to 1."


    # uniform prior input distribution r_x = Pr(X=x), column vector
    r_x = ones(channel.size_X,1)/channel.size_X
    qX_Y = similar(channel.pY_X)
    for iteration in 1:max_iter
        #qX_Y = normalize(r_x .* channel.pY_X, 1)
        qX_Y = r_x .* channel.pY_X
        qX_Y = norm_columns(r_x .* channel.pY_X)
        r_x′ =  qX_Y.^channel.pY_X
        r_x′ =  prod(qX_Y.^channel.pY_X, dims=2)

        r_x′ =  norm_columns(prod(qX_Y.^channel.pY_X, dims=2))
        residual = norm(r_x′-r_x) # p-norm defaults to p=2
        # @show iteration, residual, r_x′
        r_x = r_x′
        residual < ε && break
    end

    C = channel_capacity(channel, r_x, qX_Y)
    @show r_x, qX_Y, C
    r_x, C
end
