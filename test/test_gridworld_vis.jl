using POMDPDiscrete
using POMDPs
using POMDPModelTools
using POMDPPolicies

import POMDPDiscrete.value_iteration
using Random
using Plots

using Test

save_plots = true

@testset "succesfully creating a plot" begin
    mdp = GridWorld()
    rng = MersenneTwister(1234)
    s0 = rand(rng, initialstate(mdp))
    # test plot of the gridworld and agent location
    p = render(mdp, s=s0, title="agent location");
    @test p isa Plots.Plot
    save_plots ? savefig(p, "plots/render_agent_location.png") : nothing
end

@testset "plotting the value function as a heatmap" begin
    mdp = GridWorld(
        size=(8,8),
        absorbing_states=[State(1,1), State(4,5)],
        p_transition = 1.0,
        γ = 1.0)
    utility = reshape(value_iteration(mdp), mdp.size)
    s0 = State(3,4)
    # test plot of the gridworld and agent location
    p = render(mdp, s=s0, utility=utility, title="utility");
    @test p isa Plots.Plot
    save_plots ? savefig(p, "plots/render_utility_vector.png") : nothing
end


@testset "policy textual representation" begin
    mdp = GridWorld()
    policy = random_stochastic_policy(mdp)
    policy_text = POMDPDiscrete.policy_grid(mdp, policy)
    @test policy_text isa Matrix{String}
end

@testset "successfully creating a policy plot" begin
    mdp = GridWorld(size=(9,9))
    rng = MersenneTwister(1234)
    s0 = rand(rng, initialstate(mdp))
    # test plot of the gridworld and agent location
    random_behaviour = random_stochastic_policy(mdp)
    p = render(mdp, s=s0, policy=random_behaviour, title="policy plot");
    @test p isa Plots.Plot
    save_plots ? savefig(p, "plots/render_policy.png") : nothing
end
