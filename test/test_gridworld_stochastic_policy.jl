using POMDPDiscrete

import POMDPs.actionindex

import POMDPModelTools:ordered_actions
import POMDPModelTools:policy_transition_matrix

using Test

@testset "stochastic policy" begin
    mdp = GridWorld()
    random_policy = random_stochastic_policy(mdp)
    @test sum(random_policy.π) ≈ length(mdp.𝒮)

    uniform_policy = uniform_stochastic_policy(mdp)
    @test sum(uniform_policy.π) ≈ length(mdp.𝒮)
    for a in ordered_actions(mdp)
        i = actionindex(mdp, a)
        @test rand(uniform_policy.π[:,i]) == 1/length(mdp.𝒜)
    end
end

#=
@testset "policy_transition_matrix" begin
    mdp = GridWorld(
        size=(3,3),
        p_transition=1.0,
        absorbing_states=[State(1,1)],
        γ=1.0)
    uniform_policy = uniform_stochastic_policy(mdp)
    T = policy_transition_matrix(mdp, uniform_policy)


end
=#

