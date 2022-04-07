using POMDPDiscrete

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
    @test POMDPDiscrete.inbounds(mdp, State(1,1)) == trueHave

end


