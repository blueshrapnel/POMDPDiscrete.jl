using POMDPDiscrete
using POMDPs

import POMDPModelTools.weighted_iterator

import POMDPDiscrete.build_probabilistic_model

using Test

@testset "test probability model for 2x2 deterministic gridworld" begin
    mdp = GridWorld(
        size=(2,2),
        p_transition=1.0,
        absorbing_states=[State(1,1)],
        γ=1.0)

    P = POMDPDiscrete.build_probabilistic_model(mdp)
    # State(1,1) - absorbing so all actions remain in State(1,1)
    @test P[:, :, 1] == [1 1 1 1;
                         0 0 0 0;
                         0 0 0 0;
                         0 0 0 0]
    # State(2,1) - bottom right
    @test P[:, :, 2] == [0 0 0 1;
                         0 1 1 0;
                         0 0 0 0;
                         1 0 0 0]
    # State(1,2) - top left
    @test P[:, :, 3] == [0 0 1 0;
                         0 0 0 0;
                         1 0 0 1;
                         0 1 0 0]
    # State(2,2) - top right
    @test P[:, :, 4] == [0 0 0 0;
                         0 0 1 0;
                         0 0 0 1;
                         1 1 0 0]

end
