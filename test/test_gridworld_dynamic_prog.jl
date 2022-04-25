using POMDPDiscrete
using POMDPs

using Random

using Test

@testset "reward function" begin
    mdp=GridWorld(
        size=(4,4),
        p_transition = 1,
        absorbing_states=[State(1,1), State(1,4)],
        γ = 1
    )
    abs_sᵢs = map(s -> stateindex(mdp, s), mdp.absorbing_states)
    R = POMDPDiscrete.get_rewards(mdp)
    @test abs(sum(R)) == mdp.Nₛ - length(mdp.absorbing_states)
    for si ∈ abs_sᵢs
        @test R[si] == 0
    end
end

@testset "value_iteration" begin
    mdp=GridWorld(
        size=(4,4),
        p_transition = 1,
        absorbing_states=[State(1,1)],
        γ = 1
    )
    V = POMDPDiscrete.value_iteration(mdp)
    target = [ 0 -1 -2 -3;
              -1 -2 -3 -4;
              -2 -3 -4 -5;
              -3 -4 -5 -6]
    @test reshape(V, (4,4)) ≈ target
end

@testset "policy_evaluation" begin
    mdp = GridWorld()
    uniform_behaviour = uniform_stochastic_policy(mdp)
    V = POMDPDiscrete.policy_evaluation(mdp, uniform_behaviour)
end

@testset "initialise_vector" begin
    zero_U = POMDPDiscrete.initialise_vector(10)
    @test zero_U == zeros(10)
    rng = MersenneTwister(1234)
    random_U = POMDPDiscrete.initialise_vector(10;rng=rng)
    @test sum(random_U) != 0
end
