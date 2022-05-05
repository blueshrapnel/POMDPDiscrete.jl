
#=
Relevant information

=#

@doc raw"""

Relevant information captures the information from the world (state space) which is used directly in action selection required to achieve a given level of performance.

```math
\min_{\pi}I(S;A):\mathbb{E}[Q^\pi(s,a)] \geq \tilde{Q}
```

"""

struct InformationChannel
    size_X::Int
    size_Y::Int
    pY_gX  # transition matrix - conditional probability p_Y|X
 end


function relevant_information_policy(channel::InformationChannel, max_iter=50, ε=1e-4, log_base=2)
    pX =  ones(channel.size_X)/channel.size_X
    Z_x = similar(pX)
    pY_gX = channel.pY_gX
    Q = zeros(channel.size_X, channel.sizeY)
    k = 0
    for i ∈ ProgressBar(max_iter)
        
        # real underlying idea is to run
		# pYgX = dot( Diagonal(pY), .exp(-β * cost) ) # diagonalise pY
		# Z = sum(pYgX, dims=1)
		# pR = dot(pYgX, Diagonal(1/Z))
		# but this doesn't work when beta is really large
		# so instead we manipulate logs of p(y|x).
        
        # TODO - check whether problems arise 0 probability masses (nansum)
        pY = pYgX * pX     # marginalize to get probability distribution of output
        uYgX = log.(repeat(pY', n, 1)) - β * Q
        pYgX = exp.(uYgX)
        Z_x = sum(pYgX, dims=1)
        uYgX = log.(pYgX) - log.(repeat(Z_x, n, 1))
        pYgX = exp.(uYgX)
        pY = pYgX *  pX
    end
    return pYgX, Z_x
end