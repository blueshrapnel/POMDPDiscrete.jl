using POMDPDiscrete

using Random

using Test

@testset "policy_evaluation" begin
    mdp = GridWorld()
    uniform_behaviour = uniform_stochastic_policy(mdp)
    V = POMDPDiscrete.policy_evaluation(mdp, uniform_behaviour)
end

@testset "initialise_utility" begin
    zero_U = POMDPDiscrete.initialise_utility(10)
    @test zero_U == zeros(10)
    rng = MersenneTwister(1234)
    random_U = POMDPDiscrete.initialise_utility(10;rng=rng)
    @test sum(random_U) != 0
end
