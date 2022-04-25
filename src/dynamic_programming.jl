#= Utility functions
=#

#= POMDPModelTools.policy_reward_vector(mdp, policy rewardfunction)
iterates through successor state distribution and calculates weighted sum
of rewards over all successor states, but does not take into account
an action distribution.
=#

# return rewards as a vector of the return type specified in reward()
function get_rewards(mdp::MDP, policy=nothing)
    # don't actually use the policy
    U = map(s -> reward(mdp, s), mdp.𝒮)  # returns a matrix of R(s)
end

# returns utility as a vector
function get_utility(mdp::MDP, policy::Policy)
    U = map(s -> value(policy, s), mdp.𝒮)  # returns a matrix of V(s)
end

"""
    initialise_vector(size; rng)

Returns a vector of the size given which is filled with zeros if no random number generator is specified, if the optinal argument `rng` is supplied, it uses the generator to return a vector of random values of the given size.
"""
function initialise_vector(size::Int; rng::Union{AbstractRNG,Nothing}=nothing)
    if isnothing(rng)
        return zeros(size)
    else   # argument checked rng <: AbstractRNG
        return rand(rng, size)
    end
end

#=
Policy Evaluation
=#

@doc raw"""
    policy_evaluation(mdp, π)

Compute the value function by evaluating the policy in each state using the Bellman Expectation Equation. Sutton and Barto 2nd Edition p75
    ``` math
    V(s) = \sum_a\pi(a|s)\sum_{s^\prime, s} p(s^\prime, r |s, a) [r + \gamma v(s^\prime)]
    ```
"""
function policy_evaluation(mdp::Union{MDP,POMDP}, policy::StochasticPolicy)
    R = get_rewards(mdp)
    Nₛ = length(states(mdp))
    P = build_probabilistic_model(mdp) # only a function of the mdp
    T = reduce(hcat, [P[:, :, si] * policy.π[si, :] for si in 1:Nₛ])' # policy is given and fixed

    # initialise value function to zero
    V = initialise_vector(Nₛ)
    while true  # may want to consider a max iters approach as well
        v = copy(V)
        V = R + mdp.γ * T * v
        Δ = abs.(v - V)
        maximum(Δ) < ε && break
        #@show maximum(Δ)
    end
    return V
end


"""
    value_iteration(mdp)

Find the optimal value function for an agent acting greedily.
"""
function value_iteration(mdp::Union{MDP,POMDP})
    Nₛ = length(states(mdp))
    # build a reward matrix with indices R[s',a,s]
    R = get_rewards(mdp)  # as a column vector
    R_full = repeat(R, 1, Nₛ)
    Nₐ = length(actions(mdp))
    P = build_probabilistic_model(mdp) # only a function of the mdp
    # initialise value function to zero
    V = initialise_vector(Nₛ)
    ε = 0.0001
    while true
        qₛ = zeros(Nₐ)
        Δ = 0
        for si ∈ 1:Nₛ
            for ai ∈ 1:Nₐ
                # TODO qₛ[ai] = sum(P[:,ai,si].*(R[:, ai, si] + mdp.γ*V))
                # i.e. use R[s',a,s]
                qₛ[ai] = sum(P[:,ai,si].*(R_full[si, :] + mdp.γ*V))
            end
            best_value = maximum(qₛ)
            Δ = max(Δ, abs(V[si] - best_value))
            V[si] = best_value
        end
        Δ < ε && break
        #@show maximum(Δ)
    end
    return V
end

mdp = GridWorld()
uniform = uniform_stochastic_policy(mdp)
