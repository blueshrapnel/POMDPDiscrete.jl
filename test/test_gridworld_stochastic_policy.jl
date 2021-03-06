using POMDPDiscrete
import POMDPDiscrete.policy_transition_matrix
import POMDPDiscrete.action_distribution

using POMDPs
import POMDPModelTools.SparseCat
import POMDPModelTools:ordered_actions

using Random

using Test

@testset "stochastic policy" begin
    mdp = GridWorld()
    random_policy = random_stochastic_policy(mdp)
    @test sum(random_policy.Ï) â length(mdp.ğ®)

    uniform_policy = uniform_stochastic_policy(mdp)
    @test sum(uniform_policy.Ï) â length(mdp.ğ®)
    for a â ordered_actions(mdp)
        i = actionindex(mdp, a)
        @test rand(uniform_policy.Ï[:,i]) == 1/length(mdp.ğ)
    end
end

@testset "policy_transition_matrix - 2x2 grid absorbing" begin
    mdp = GridWorld(
        size=(2,2),
        p_transition=1.0,
        absorbing_states=[State(1,1)],
        Î³=1.0)
    policy = uniform_stochastic_policy(mdp)
    T = policy_transition_matrix(mdp, policy)
    @test T == [1    0    0    0;
                0.25 0.5  0    0.25;
                0.25 0    0.5  0.25;
                0    0.25 0.25 0.5]
    # check successor state distribution sums to 1
    @test all(sum(T, dims=2) .â 1)
end

@testset "policy_transition_matrix - stochastic grid" begin
    mdp = GridWorld(
        size=(3,7),
        p_transition=0.6,
        absorbing_states=State[],
        Î³=1.0)
    policy = random_stochastic_policy(mdp)
    T = policy_transition_matrix(mdp,policy)
    # expected value
    Nâ = length(states(mdp))
    Nâ = length(actions(mdp))
    Tâ = zeros(Nâ, Nâ)
    P = POMDPDiscrete.build_probabilistic_model(mdp)
    for si â 1:Nâ
        for sâ²i â 1:Nâ
            for ai â 1:Nâ
                Tâ[si, sâ²i] += policy.Ï[si, ai] * P[sâ²i, ai, si]
            end
        end
    end
    @test T â Tâ
    # check successor state distribution sums to 1
    @test all(sum(T, dims=2) .â 1)
end


@testset "action_distribution and sampling methods" begin
    mdp = GridWorld(
        size=(3,7),
        p_transition=0.6,
        absorbing_states=State[],
        Î³=1.0)
    policy = random_stochastic_policy(mdp)
    rng = MersenneTwister(1234)
    s0 = rand(rng, initialstate(mdp))
    a_dist = action_distribution(policy, s0)
    @test typeof(a_dist) <: SparseCat{<:AbstractArray, <:AbstractArray}
    @test a_dist.vals == ordered_actions(mdp)
    @test sum(a_dist.probs) â 1
end
