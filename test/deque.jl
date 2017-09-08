@testset "Push, pop, shift, unshift, etc" begin
    v = @SVector [1, 2, 3]

    @test @inferred(push(v, 4)) === @SVector [1, 2, 3, 4]
    @test @inferred(pop(v)) === @SVector [1, 2]

    @test @inferred(unshift(v, 0)) === @SVector [0, 1, 2, 3]
    @test @inferred(shift(v)) === @SVector [2, 3]

    @test @inferred(insert(v, 2, -2)) === @SVector [1, -2, 2, 3]
    @test @inferred(deleteat(v, 2)) === @SVector [1, 3]

    @test @inferred(setindex(v, -2, 2)) === @SVector [1, -2, 3]

    @test_throws BoundsError insert(v, -2, 2)
    @test_throws BoundsError deleteat(v, -2)
    @test_throws BoundsError setindex(v, 2, -2)
end

@testset "setindex" begin
    a = @MArray ones(3, 3, 3)
    a[2, 1, 3] = 2.
    @test a[2, 1, 3] == 2.
end
