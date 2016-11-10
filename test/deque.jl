@testset "Push, pop, shift, unshift, etc" begin
    v = @SVector [1, 2, 3]

    @test push(v, 4) === @SVector [1, 2, 3, 4]
    @test pop(v) === @SVector [1, 2]

    @test unshift(v, 0) === @SVector [0, 1, 2, 3]
    @test shift(v) === @SVector [2, 3]

    @test (@inferred insert(v, 2, -2)) === @SVector [1, -2, 2, 3]
    @test (@inferred deleteat(v, 2)) === @SVector [1, 3]

    @test setindex(v, -2, 2) === @SVector [1, -2, 3]
end
