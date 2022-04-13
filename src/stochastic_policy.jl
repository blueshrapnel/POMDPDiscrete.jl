
#=
POMDPs includes a function to return a random number generator `RandomPolicy(mdp)``, iterally a random action every time.  For a stationary Stochastic policy we want a constant action distribution given the state.
=#
struct StochasticPolicy{P<:Union{POMDP, MDP}, T<:AbstractMatrix{Float64}, A} <: Policy
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
