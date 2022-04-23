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
    U = map(s->reward(mdp, s), mdp.𝒮)  # returns a matrix of R(s)
end

# returns utility as a vector
function get_utility(mdp::MDP, policy::Policy)
    U = map(s->value(policy, s), mdp.𝒮)  # returns a matrix of V(s)
end

"""
    initialise_utility(size; rng)

Returns a vector of the size given which is filled with zeros if no random number generator is specified, if the optinal argument `rng` is supplied, it uses the generator to return a vector of random values of the given size.
"""
function initialise_utility(size::Int; rng::Union{AbstractRNG, Nothing}=nothing)
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
function policy_evaluation(mdp::Union{MDP, POMDP}, policy::StochasticPolicy)
    R = Array(get_rewards(mdp))
    Nₛ = length(states(mdp))
    P = build_probabilistic_model(mdp) # only a function of the mdp
    T = reduce(hcat, [P[:,:,si]*policy.π[si,:] for si in 1:Nₛ])' # policy is given and fixed

    # initialise value function to zero
    V = initialise_utility(Nₛ)
    while true  # may want to consider a max iters approach as well
        v = copy(V)
        V = R + mdp.γ*T*v
        Δ = abs.(v - V)
        maximum(Δ) < ε && break
        #@show maximum(Δ)
    end
    return V
end


function one_step_lookahead(state, value)

end


function value_iteration(mdp::Union{MDP, POMDP})


end

mdp = GridWorld()
uniform = uniform_stochastic_policy(mdp)
