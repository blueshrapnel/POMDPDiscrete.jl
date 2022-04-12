using POMDPDiscrete
using POMDPs
using POMDPModelTools

using Random
using Plots

using Test

@testset "succesfully creating a plot" begin
    mdp = GridWorld()
    rng = MersenneTwister(1234)
    s0 = rand(rng, initialstate(mdp))
    # test plot of the gridworld and agent location 
    p = render(mdp, s=s0);
    @test p isa Plots.Plot
    savefig(p, "test_render_agent_location.png")  
end