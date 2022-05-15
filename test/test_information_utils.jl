using POMDPDiscrete
import POMDPDiscrete: xlog2x, xlog2y
import POMDPDiscrete: entropy, conditional_entropy

using Test

@testset "xlog2x and xlog2y functions" begin
    @test xlog2x.([0, 0.5, 8, 1e-18]) ≈ [0, -0.5, 24, -5.979470570797254e-17]
    @test xlog2y.([1, 2, 3, 4], [0, 0.5, 8, 1e-18]) ≈ [0, -2, 9, -239.1788228318901]
end

@testset "entropy rationals with zero" begin
    pX = [1//2, 1//4, 1//8, 1//8, 0]
    HX = entropy(pX)
    @test HX ≈ 7//4   # rationals
    @test entropy([1/2, 1/4, 1/8, 1/8, 0]) ≈ 7/4 # floats
end


@testset "entropy and conditional entropy" begin
    # example similar to Thomas and Cover p17-18, changed to ensure not symmetrical
    # joint distribution with X indexed by column, Y indexed by row
    pXY = [[1// 8 1//16 1//32 1//32];
           [1//16 1//8  1//32 1//32];
           [1//16 1//16 1//32 1//16];
           [1//4  0     1//32 0    ]]

    pX = sum(pXY, dims=1)
    pY = sum(pXY, dims=2)
    pY_X = pXY./pX
    pX_Y = pXY./pY

    HX = entropy(pX)
    @test HX ≈ 7/4

    HY = entropy(pY)
    @test HY ≈ 1.994349704
    # test conditional entropy
    HY_X = conditional_entropy(pY_X, pX)
    HX_Y = conditional_entropy(pX_Y', pY)
    @test HY_X ≈ 27/16
    @test HX_Y ≈ 1.443150296
    # test H(X,Y) = H(X) + H(Y|X)
    @test HX + HY_X ≈ HY + HX_Y

end
