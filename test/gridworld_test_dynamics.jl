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

@testset "GridWorld Constructor and discount" begin
    @test GridWorld() isa GridWorld
    @test GridWorld(
        size=(4,4), p_transition=0.8, 
        absorbing_states=[State(1,1), State(4,4)], 
        γ=0.99) isa GridWorld
    mdp = GridWorld(γ=0.5)
    @test discount(mdp) == 0.5
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
        absorbing_states=[State(1,1), State(4,4)],
        p_transition=0.7)
    @test reward(mdp, State(1,1)) == 0
    @test reward(mdp, State(4,4)) == 0
    @test reward(mdp, State(1,2)) == -1
    @test reward(mdp, State(3,2)) == -1

    @test isterminal(mdp, State(1,1))
    @test isterminal(mdp, State(4,4))
    @test !isterminal(mdp, State(4,3))
    @test !isterminal(mdp, State(2,3))
end

@testset "transition and next state distribution" begin
    mdp = GridWorld(
        size=(9,11),
        absorbing_states=[State(5,6)])
    s = rand(initialstate(mdp))
    p_s′ = transition(mdp, s, :up)

    if p_s′ isa SparseCat
        @test sum(p_s′.probs)  ≈ 1
    elseif p_s′ isa Deterministic
        @test pdf(p_s′, p_s′.val) == 1
    else
        # currently not returning any other distributions    
        @test false
    end

    # now test specific spcecific transitions
    # confirming that bottom left is State(1,1)
    s = State(3,3)
    p_s′ = transition(mdp, s, :down)
    @test pdf(p_s′, State(3,2)) ≈ 0.7
    @test pdf(p_s′, State(2,3)) ≈ 0.1
    @test pdf(p_s′, State(4,3)) ≈ 0.1
    @test pdf(p_s′, State(3,4)) ≈ 0.1
    s = mode(p_s′)
    s′ = mode(transition(mdp, s, :right))
    @test s′ == State(4,2)
    s′′ = mode(transition(mdp, s′, :down))
    @test s′′ == State(4,1)
    # testing transition on the boundary
    p_s′′′ = transition(mdp, s′′, :down)
    @test pdf(p_s′′′, State(4,1)) ≈ 0.7
    @test pdf(p_s′′′, State(3,1)) ≈ 0.1
    @test pdf(p_s′′′, State(5,1)) ≈ 0.1
    @test pdf(p_s′′′, State(4,2)) ≈ 0.1

    # testing transition into absorbing state
    s = State(6,5)
    s′ = mode(transition(mdp, s, :up))
    @test s′ == State(6,6)
    p_s′′ = transition(mdp, s′, :left)
    @test pdf(p_s′′, State(5,6)) ≈ 0.7
    @test pdf(p_s′′, State(6,7)) ≈ 0.1
    @test pdf(p_s′′, State(6,5)) ≈ 0.1
    @test pdf(p_s′′, State(7,6)) ≈ 0.1

    # test all actions from terminal state absorbed
    abs_s = State(5,6)
    for a in actions(mdp)
        p_s′ = transition(mdp, abs_s, a) 
        @test p_s′ isa Deterministic
        @test mode(p_s′) == State(5,6)
    end

    # test corner state
    s = State(1,1)
    p_s′ = transition(mdp, s, :left)
    @test pdf(p_s′, State(1,1)) ≈ 0.8
    @test pdf(p_s′, State(2,1)) ≈ 0.1
    @test pdf(p_s′, State(1,2)) ≈ 0.1
    # test a random other state is zero
    @test pdf(p_s′, State(5,6)) ≈ 0
end