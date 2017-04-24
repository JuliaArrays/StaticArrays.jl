@testset "SUnitRange" begin
    @test length(StaticArrays.SUnitRange(1,3)) === 3
    @test length(StaticArrays.SUnitRange(1,-10)) === 0

    @test StaticArrays.SUnitRange(2,4)[2] === 3
end
