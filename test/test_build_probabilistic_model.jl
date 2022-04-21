using POMDPDiscrete
using POMDPs

import POMDPModelTools.weighted_iterator

import POMDPDiscrete.build_probabilistic_model

using Test

@testset "test 2x2 deterministic gridworld - State(1,1) absorbing" begin
    mdp = GridWorld(
        size=(2,2),
        p_transition=1.0,
        absorbing_states=[State(1,1)],
        γ=1.0)

    P = POMDPDiscrete.build_probabilistic_model(mdp)
    # State(1,1) - absorbing so all actions remain in State(1,1)
    @test P[:, :, 1] ≈ [1 1 1 1;
                         0 0 0 0;
                         0 0 0 0;
                         0 0 0 0]
    # State(2,1) - bottom right
    @test P[:, :, 2] ≈ [0 0 0 1;
                         0 1 1 0;
                         0 0 0 0;
                         1 0 0 0]
    # State(1,2) - top left
    @test P[:, :, 3] ≈ [0 0 1 0;
                         0 0 0 0;
                         1 0 0 1;
                         0 1 0 0]
    # State(2,2) - top right
    @test P[:, :, 4] ≈ [0 0 0 0;
                         0 0 1 0;
                         0 0 0 1;
                         1 1 0 0]

end

@testset "test 2x2 deterministic gridworld - no terminal state" begin
    mdp = GridWorld(
        size=(2,2),
        p_transition=1.0,
        absorbing_states=State[],
        γ=1.0)

    P = POMDPDiscrete.build_probabilistic_model(mdp)
    # State(1,1) - bottom left
    @test P[:, :, 1] ≈ [0 0 1 1;
                         0 1 0 0;
                         1 0 0 0;
                         0 0 0 0]
    # State(2,1) - bottom right
    @test P[:, :, 2] ≈ [0 0 0 1;
                         0 1 1 0;
                         0 0 0 0;
                         1 0 0 0]
    # State(1,2) - top left
    @test P[:, :, 3] ≈ [0 0 1 0;
                         0 0 0 0;
                         1 0 0 1;
                         0 1 0 0]
    # State(2,2) - top right
    @test P[:, :, 4] ≈ [0 0 0 0;
                         0 0 1 0;
                         0 0 0 1;
                         1 1 0 0]

end

@testset "test 2x2 stochastic (0.7) gridworld - no terminal state" begin
    mdp = GridWorld(
        size=(2,2),
        p_transition=0.7,
        absorbing_states=State[],
        γ=1.0)

    P = POMDPDiscrete.build_probabilistic_model(mdp)
    # State(1,1) - bottom left
    @test P[:, :, 1] ≈ [0.2 0.2 0.8 0.8;
                         0.1 0.7 0.1 0.1;
                         0.7 0.1 0.1 0.1;
                         0   0   0   0  ]
    # State(2,1) - bottom rig
    @test P[:, :, 2] ≈ [0.1 0.1 0.1 0.7;
                         0.2 0.8 0.8 0.2;
                         0   0   0   0  ;
                         0.7 0.1 0.1 0.1]
    # State(1,2) - top left
    @test P[:, :, 3] ≈ [0.1 0.1 0.7 0.1;
                         0   0   0   0  ;
                         0.8 0.2 0.2 0.8;
                         0.1 0.7 0.1 0.1]
    # State(2,2) - top right
    @test P[:, :, 4] ≈ [0   0   0   0  ;
                         0.1 0.1 0.7 0.1;
                         0.1 0.1 0.1 0.7;
                         0.8 0.8 0.2 0.2]

end

@testset "transitions summed over all action = 1" begin
    mdp = GridWorld(
        size=(4,5),
        p_transition=0.7,
        absorbing_states=State[State(1,1)],
        γ=1.0)

    P = POMDPDiscrete.build_probabilistic_model(mdp)

    for a  in 1:mdp.Nₐ
        @test all(sum(P[:,:,a], dims=1) .≈ 1)
    end
end
