using POMDPDiscrete
import POMDPDiscrete: InformationChannel
import POMDPDiscrete: blahut_arimoto_channel_capacity

using Test

@testset "binary symmetric channel" begin
    e = 0.2
    binary_symmetric_channel = InformationChannel(
    # p = 0.2
    2, 2, [1-e e; e 1-e]
    )
    r_x, C = blahut_arimoto_channel_capacity(binary_symmetric_channel)
    # The analytic solution of the capaciy 1-(H_Pₑ) with r_x = [0.5; 0.5]
    H_Pₑ = -e * log(2,e) - (1-e) * log(2, 1-e)
    @test C ≈ 1 - H_Pₑ

end

@testset "noiseless binary symmetric channel" begin
    noiseless_binary_symmetric_channel = InformationChannel(
   2, 2, [1 0; 0 1]
    )
    r_x, C = blahut_arimoto_channel_capacity(noiseless_binary_symmetric_channel)
    @test C ≈ 1
end

@testset "erasure channel" begin
    e = 0.2
    erasure_channel = InformationChannel(2, 3, [1-e e 0; 0 e 1-e])
    r_x, C = blahut_arimoto_channel_capacity(erasure_channel)
    @test C ≈ 1-e
end

@testset "binary biased channel" begin
    binary_biased_channel = InformationChannel(
   2, 2, [1 0; 0.5 0.5]
    )
    r_x, C = blahut_arimoto_channel_capacity(binary_biased_channel)
    @test isapprox(C, 0.3219, rtol=1e-3)
end

@testset "trip channel" begin
    trip_channel = InformationChannel(
   3,3, [0.6 0.3 0.1; 0.7 0.1 0.2; 0.5 0.05 0.45]
    )
    r_x, C = blahut_arimoto_channel_capacity(trip_channel)
    # @test r_x ≈ [0.501735; 0; 0.498265]
    # @test C ≈ 0.161631
    # getting the right ballpark - find source to get exact answers
end
