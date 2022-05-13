
"""
    StochasticPolicy(mdp, policy_matrix)

Why do we want a stochastic policy?  Acting optimally in all situations comes with a cost, a stochastic policy allows the agent to save information bandwidth and/or storage by accepting some loss of optimality in some situations

POMDPs includes a function to return a random number generator `RandomPolicy(mdp)``, iterally a random action every time.  For a stationary Stochastic policy we want a constant action distribution given the state.

    Policy abstract type defined in POMDPs.jl/src/policy.jl.  Implement
        action, updater, value.
"""
struct StochasticPolicy{P<:Union{POMDP, MDP}, T<:AbstractMatrix{<:Real}, A} <: Policy
	mdp::P
	π::T
	act::Vector{A}
end

function StochasticPolicy(
	mdp::Union{POMDP,MDP},
	π=ones(length(states(mdp)), length(actions(mdp)))
	)
	# normalise the state_action_dist ∑_a π(a|s) = 1
	π = π./sum(π, dims=2)
	return StochasticPolicy(mdp, π, ordered_actions(mdp))
end


"""
    action_distribution(policy::StochasticPolicy, x)

Returns the distribution of the ordered actions given the policy and the current state or belief x.
"""
function action_distribution(policy::StochasticPolicy, x)
    xi = stateindex(policy.mdp, x)
    return  SparseCat(policy.act, policy.π[xi,:])
end

"""
    action(policy::StocasticPolicy, x)

Return an action sampled from the action distribution given the policy and the state `x`.
"""

function action(policy::StochasticPolicy, x::State)
    return rand(MersenneTwister(), action_distribution(policy, x))
end

"""
    uniform_stochastic_policy(mdp)

Return a stationary `StochasticPolicy` where all actions are equally likely for each state.
"""
function uniform_stochastic_policy(mdp::Union{POMDP, MDP})
    return StochasticPolicy(mdp)
end

"""
    random_stochastic_policy(mdp)

Return a stationary `StochasticPolicy`` where the action distribution for each state is random.
"""
function random_stochastic_policy(mdp::Union{POMDP, MDP})
	π = rand(length(states(mdp)), length(actions(mdp)))
	return StochasticPolicy(mdp, π)
end

"""
    policy_transition_matrix(mdp, π)

Return a transition matrix given a probabilistic model P[s′, a, s] (calculated using mdp dynamics)) and a policy stochastic π[s, a].
"""
function policy_transition_matrix(mdp::Union{POMDP,MDP}, π::Matrix{<:Real})
    Nₛ = length(states(mdp))
    P = build_probabilistic_model(mdp)
    return reduce(hcat, [P[:,:,si]*π[si,:] for si in 1:Nₛ])'
end

function policy_transition_matrix(mdp::Union{POMDP,MDP}, policy::StochasticPolicy)
    return policy_transition_matrix(mdp, policy.π)
end
