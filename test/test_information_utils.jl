using POMDPDiscrete
import POMDPDiscrete: xlog2x, xlog2y
import POMDPDiscrete: compute_entropy

using Test

@testset "xlog2x and xlog2y functions" begin
    @test xlog2x.([0, 0.5, 8, 1e-18]) ≈ [0, -0.5, 24, -5.979470570797254e-17]
    @test xlog2y.([1, 2, 3, 4], [0, 0.5, 8, 1e-18]) ≈ [0, -2, 9, -239.1788228318901]
end


@testset "computing entropy" begin
    @test compute_entropy([1//2, 1//4, 1//8, 1//8, 0]) ≈ 7//4
    
end
