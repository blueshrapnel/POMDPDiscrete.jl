using POMDPDiscrete:State
using Test

@testset "state_tests.jl" begin
    # test adding states to effect actions
    @test State(0,1) + State(1,0) == State(1,1)
    @test State(0,1) + State(0,0) == State(0,1)
    @test State(0,0) + State(-1,-1) == State(-1,-1)

    # test equivalent states
    @test State(5,5) == State(5,5)
    @test State(4,3) != State(0,0)
end