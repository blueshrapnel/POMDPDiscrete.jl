using POMDPDiscrete

import POMDPModelTools:ordered_actions
import POMDPs.actionindex

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
