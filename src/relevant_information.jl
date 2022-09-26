"""
Relevant information

What is the minimum level of information to act at a certain level?  We can also answer the
question of what that action looks like.  We view the perception-action loop as an information
channel.  Information flows from the environment to the agent via sensors, and from the agent
to the environment through actuators.

"""

@doc raw"""

Relevant information captures the information from the world (state space) which is used directly in action selection required to achieve a given level of performance.

```math
\min_{\pi}I(S;A):\mathbb{E}[Q^\pi(s,a)] \geq \tilde{Q}
```

"""


function InformationChannel(mdp::MDP)
    # defaults to uniform conditional distribution
    size_X = length(states(mdp))
    size_Y = length(actions(mdp))
    pY_X = uniform_stochastic_policy(mdp).π
    return InformationChannel(size_X, size_Y, pY_X)
end


function relevant_information_policy(channel::InformationChannel, mdp::MDP; β=10, max_iters=50, ε=1e-4)
    Nₓ = channel.size_X
    pX =  ones(Nₓ)/Nₓ
    Z_x = similar(pX)
    pY_X = channel.pY_X
    Q = zeros(Nₓ, channel.size_Y)
    Δ = 0
    for i ∈ 1:max_iters
        # TODO - check whether problems arise 0 probability masses Nan, Inf, etc.
        # TODO - move blahut arimoto step into separate function
        # TODO - can we refine a Q instead of initialising to zero?
        pY = pY_X' * pX     # marginalize to get probability distribution of output

        # for higher precision use BigFloat
        convert(Array{BigFloat}, Q)

        # if Qs get too large, then manipulate logs of β Q
        # uY_X = log.(repeat(pY', Nₓ, 1)) + β * Q
        # pY_X = exp.(uY_X)
        # Z_x = sum(pY_X, dims=2)
        # uY_X = log.(pY_X) - log.(repeat(Z_x, Nₓ, 1))
        # pY_X = exp.(uY_X)

        # instead of repeat could use diagonal
        # pY_X = dot( Diagonal(pY), .exp(β * Q) ) # diagonalise pY
        pY_X = repeat(pY', Nₓ, 1) .* exp2.(β * Q)
        Z_x = sum(pY_X, dims=2)

        pY_X = pY_X ./ Z_x


        # update the Q function
        Q_old = copy(Q)
        update_state_action_value!(mdp, pY_X, Q)
        Δ = norm(Q - Q_old)         # p-norm defaults to p=2
        Δ < ε && break

        #=  TODO - check for convergence using policy as well?
        Δ = DKL between two consecutive policies
        =#
    end

    remove_negligible_values_normalise!(pY_X) # then renormalise
    return pY_X, Q, Z_x
end

# neglible threshold specifed in parameters.jl
function remove_negligible_values_normalise!(values, threshold=negligible_threshold)
    indices = findall(x->x<threshold, values)
    values[indices] .= 0
    values ./= sum(values, dims=2)
end
