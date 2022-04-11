using POMDPDiscrete
using POMDPs
using POMDPModelTools

using Test

@testset "state addition and comparison" begin
    # test adding states to effect actions
    @test State(0,1) + State(1,0) == State(1,1)
    @test State(0,1) + State(0,0) == State(0,1)
    @test State(0,0) + State(-1,-1) == State(-1,-1)

    # test equivalent states
    @test State(5,5) == State(5,5)
    @test State(4,3) != State(0,0)
end

@testset "GridWorld Constructor" begin
    @test GridWorld() isa GridWorld
    @test GridWorld(
        size=(4,4), p_transition=0.8, 
        absorbing_states=[State(1,1), State(4,4)], 
        γ=0.99) isa GridWorld
        @test GridWorld(γ=0.5) isa GridWorld
end

function test_state_indexing(mdp::GridWorld, ss::Vector{State}) 
    for (i,s) in enumerate(states(mdp))
        if s != ss[i]
            return false
        end
    end
    return true
end

@testset "state space for a 5x7 grid world" begin
    # create a default 5x7 GridWorld mdp
    mdp = GridWorld()

    # test inbounds function
    @test POMDPDiscrete.inbounds(mdp, State(3,4)) == true
    @test POMDPDiscrete.inbounds(mdp, State(8,4)) == false
    @test POMDPDiscrete.inbounds(mdp, State(3,9)) == false
    @test POMDPDiscrete.inbounds(mdp, State(9,9)) == false
    @test POMDPDiscrete.inbounds(mdp, State(0,0)) == false
    @test POMDPDiscrete.inbounds(mdp, State(5,7)) == true
    @test POMDPDiscrete.inbounds(mdp, State(1,1)) == true

    state_iterator = states(mdp)
    ss = ordered_states(mdp)
    @test length(ss) == length(mdp)
    @test test_state_indexing(mdp, ss)

    @test rand(initialstate(mdp)) isa State

end

function test_action_indexing(mdp::GridWorld, action_space::Vector{Symbol}) 
    for (i,a) in enumerate(ordered_actions(mdp))
        if a != action_space[i]
            return false
        end
    end
    return true
end

@testset "action space" begin
    mdp = GridWorld()
    action_space = actions(mdp)
    @test action_space == ordered_actions(mdp)
    @test test_action_indexing(mdp, action_space)
end

@testset "reward" begin
    mdp = GridWorld(
        size=(4,4),
        absorbing_states=[State(1,1), State(4,4)])
    @test reward(mdp, State(1,1)) == 0
    @test reward(mdp, State(4,4)) == 0
    @test reward(mdp, State(1,2)) == -1
    @test reward(mdp, State(3,2)) == -1

    @test isterminal(mdp, State(1,1))
    @test isterminal(mdp, State(4,4))
    @test !isterminal(mdp, State(4,3))
    @test !isterminal(mdp, State(2,3))
end

