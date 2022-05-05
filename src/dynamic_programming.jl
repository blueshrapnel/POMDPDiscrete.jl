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
    R = get_rewards(mdp)                    # as a column vector
    R_s_s′= repeat(R, 1, Nₛ)                # R[s, s']
    Nₐ = length(actions(mdp))
    P = build_probabilistic_model(mdp)      # only a function of the mdp
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
                qₛ[ai] = sum(P[:,ai,si].*(R_s_s′[si, :] + mdp.γ*V))
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
"""
    one_step_lookahead(si, P, R_s_s′, γ, V, Nₐ)

Returns qₛ[ai] for the given state index `si`.  Requires as arguments the probabilistic model `P[s′, a, s]`, the full reward matrix `R[s, s′]`, the discount factor γ and the value function `V`.

Used as a helper function for `value iteration` and `greedy_policy`.
"""
function one_step_lookahead(si, P, R_s_s′, γ, V, Nₐ)
    qₛ = zeros(Nₐ)
    for ai ∈ 1:Nₐ
        # TODO qₛ[ai] = sum(P[:,ai,si].*(R[:, ai, si] + mdp.γ*V))
        # i.e. use R[s',a,s]
        qₛ[ai] = sum(P[:,ai,si].*(R_s_s′[si, :] + γ*V))
    end
    return qₛ
end

# given the state action value for a state, return the best value and actions
function one_step_lookahead_actions(qₛ)
    best_value = maximum(qₛ)
    #best_actions = findall(qₛ .≈ best_value)
    best_actions = findall(qₛ .≈ best_value)
    return best_actions
end

# wrapper function to return the best_value
function one_step_lookahead_value(si, P, R_s_s′, γ, V, Nₐ)
    qₛ = one_step_lookahead(si, P, R_s_s′, γ, V, Nₐ)
    best_value = maximum(qₛ)
    return best_value
end

# wrapper function to return the best_value and best actions
function one_step_lookahead_value_actions(si, P, R_s_s′, γ, V, Nₐ)
    qₛ = one_step_lookahead(si, P, R_s_s′, γ, V, Nₐ)
    best_value = maximum(qₛ)
    best_actions = one_step_lookahead_actions(qₛ)
    return best_value, best_actions
end

@doc raw"""
greedy_policy(mdp, value)

Return the deterministic optimal policy ``\pi \approx \pi^*`` such that ``\pi(s) = \arg \max_a \sum_{s', r} p(s',r|s,a)[r + \gamma V(s')]``.
"""
function greedy_policy(mdp::Union{MDP, POMDP}, V)
    Nₛ = length(states(mdp))
    Nₐ = length(actions(mdp))
    R = get_rewards(mdp)                    # as a column vector
    R_s_s′= repeat(R, 1, Nₛ)                # R[s, s']
    Nₐ = length(actions(mdp))
    P = build_probabilistic_model(mdp)      # only a function of the mdp

    π = zeros(Nₛ, Nₐ)
    for si ∈ 1:Nₛ
        qₛ =  one_step_lookahead(si, P, R_s_s′, mdp.γ, V, Nₐ)
        best_actions = one_step_lookahead_actions(qₛ)
        for ai in best_actions
            π[si, ai] = 1.0/length(best_actions)
        end
    end
    return π
end


function update_state_action_value!(mdp, π, Q)
    # Q: assume columns are action (output), rows are states (input)
    # π: assume columns are action (output), rows are states (input)  
    Nₛ, Nₐ = size(π)
    for si ∈ 1:Nₛ
        for ai ∈ 1:Nₐ
            ps′ = transition(mdp, si, ai)
            for (s′, p) in weighted_iterator(ps′)
                if p > 0.0
                    s′i = stateindex(mdp, s′)
                    q_si_s′i = 
                end    
                Q[si, ai] = sum(P[:,ai,si].*(R_s_s′[si, :] + mdp.γ*V))
            end
        end
        best_value = maximum(qₛ)
        Δ = max(Δ, abs(V[si] - best_value))
        V[si] = best_value
    end

end