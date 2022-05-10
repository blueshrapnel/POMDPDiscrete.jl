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
    pYgX  # conditional probability p_Y|X
 end


 function InformationChannel(mdp::MDP)
    size_X = length(states(mdp))
    size_Y = length(actions(mdp))
    pYgX = uniform_stochastic_policy(mdp).π
    return InformationChannel(size_X, size_Y, pYgX)
 end


function relevant_information_policy(channel::InformationChannel, mdp::MDP; β=10, max_iters=50, ε=1e-4, log_base=2)
    Nₓ = channel.size_X
    pX =  ones(Nₓ)/Nₓ
    Z_x = similar(pX)
    pYgX = channel.pYgX
    Q = zeros(Nₓ, channel.size_Y)
    Δ = 0
    for i ∈ 1:max_iters

        # real underlying idea is to run
		# pYgX = dot( Diagonal(pY), .exp(-β * cost) ) # diagonalise pY
		# Z = sum(pYgX, dims=1)
		# pR = dot(pYgX, Diagonal(1/Z))
		# but this doesn't work when beta is really large
		# so instead we manipulate logs of p(y|x).

        # TODO - check whether problems arise 0 probability masses
        # TODO - move blahut arimoto step into separate function
        pY = pYgX' * pX     # marginalize to get probability distribution of output

        # for higher precision use BigFloat
        # convert(Array{BigFloat}, Q)

        # if Qs get too large, then work through log Q
        # uYgX = log.(repeat(pY', Nₓ, 1)) + β * Q
        # pYgX = exp.(uYgX)
        pYgX = repeat(pY', Nₓ, 1) .* exp.(β * Q)
        Z_x = sum(pYgX, dims=2)
        # uYgX = log.(pYgX) - log.(repeat(Z_x, Nₓ, 1))
        # pYgX = exp.(uYgX)
        pYgX = pYgX ./ Z_x


        # update the Q function
        update_state_action_value!(mdp, pYgX, Q)

        #=  TODO - check for convergence instead of only #iterations
        Δ = DKL between two consecutive policies
        Δ < ε && break
        =#
    end

    remove_negligible_values!(pYgX)
    return pYgX, Z_x
end

function remove_negligible_values!(values, threshold=1e-6)
    indices = findall(x->x<threshold, values)
    values[indices] .= 0
end
