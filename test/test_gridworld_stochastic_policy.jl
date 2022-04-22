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
    @test sum(random_policy.π) ≈ length(mdp.𝒮)

    uniform_policy = uniform_stochastic_policy(mdp)
    @test sum(uniform_policy.π) ≈ length(mdp.𝒮)
    for a ∈ ordered_actions(mdp)
        i = actionindex(mdp, a)
        @test rand(uniform_policy.π[:,i]) == 1/length(mdp.𝒜)
    end
end

@testset "policy_transition_matrix - 2x2 grid absorbing" begin
    mdp = GridWorld(
        size=(2,2),
        p_transition=1.0,
        absorbing_states=[State(1,1)],
        γ=1.0)
    policy = uniform_stochastic_policy(mdp)
    T = policy_transition_matrix(mdp, policy)
    @test T == [1    0    0    0;
                0.25 0.5  0    0.25;
                0.25 0    0.5  0.25;
                0    0.25 0.25 0.5]
    # check successor state distribution sums to 1
    @test all(sum(T, dims=2) .≈ 1)
end

@testset "policy_transition_matrix - stochastic grid" begin
    mdp = GridWorld(
        size=(3,7),
        p_transition=0.6,
        absorbing_states=State[],
        γ=1.0)
    policy = random_stochastic_policy(mdp)
    T = policy_transition_matrix(mdp,policy)
    # expected value
    Nₛ = length(states(mdp))
    Nₐ = length(actions(mdp))
    T₂ = zeros(Nₛ, Nₛ)
    P = POMDPDiscrete.build_probabilistic_model(mdp)
    for si ∈ 1:Nₛ
        for s′i ∈ 1:Nₛ
            for ai ∈ 1:Nₐ
                T₂[si, s′i] += policy.π[si, ai] * P[s′i, ai, si]
            end
        end
    end
    @test T ≈ T₂
    # check successor state distribution sums to 1
    @test all(sum(T, dims=2) .≈ 1)
end


@testset "action_distribution and sampling methods" begin
    mdp = GridWorld(
        size=(3,7),
        p_transition=0.6,
        absorbing_states=State[],
        γ=1.0)
    policy = random_stochastic_policy(mdp)
    rng = MersenneTwister(1234)
    s0 = rand(rng, initialstate(mdp))
    a_dist = action_distribution(policy, s0)
    @test typeof(a_dist) <: SparseCat{<:AbstractArray, <:AbstractArray}
    @test a_dist.vals == ordered_actions(mdp)
    @test sum(a_dist.probs) ≈ 1
end
