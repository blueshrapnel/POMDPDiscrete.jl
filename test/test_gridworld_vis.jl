using POMDPDiscrete
using POMDPs
using POMDPModelTools
using POMDPPolicies

using Random
using Plots

using Test

save_plots = false

@testset "succesfully creating a plot" begin
    mdp = GridWorld()
    rng = MersenneTwister(1234)
    s0 = rand(rng, initialstate(mdp))
    # test plot of the gridworld and agent location
    p = render(mdp, s=s0);
    @test p isa Plots.Plot
    save_plots ? savefig(p, "test/test_render_agent_location.png") : nothing
end

@testset "policy textual representation" begin
    mdp = GridWorld()
    policy = POMDPPolicies.RandomPolicy(mdp)
    @test POMDPDiscrete.policy_grid(mdp, policy) isa Matrix{String}
end

@testset "successfully creating a policy plot" begin
    mdp = GridWorld(size=(9,9))
    rng = MersenneTwister(1234)
    s0 = rand(rng, initialstate(mdp))
    # test plot of the gridworld and agent location
    random_behaviour = random_stochastic_policy(mdp)
    p = render(mdp, s=s0, policy=random_behaviour);
    @test p isa Plots.Plot
    save_plots ? savefig(p, "test/test_render_policy.png") : nothing
end
