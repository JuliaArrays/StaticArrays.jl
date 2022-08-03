using StaticArrays, Test

@testset "Push, pop, pushfirst, popfirst, etc" begin
    v = @SVector [1, 2, 3]

    @test @inferred(push(v, 4)) === @SVector [1, 2, 3, 4]
    @test @inferred(pop(v)) === @SVector [1, 2]

    @test @inferred(pushfirst(v, 0)) === @SVector [0, 1, 2, 3]
    @test @inferred(popfirst(v)) === @SVector [2, 3]

    @test @inferred(insert(v, 2, -2)) === @SVector [1, -2, 2, 3]
    @test @inferred(deleteat(v, 2)) === @SVector [1, 3]

    @test @inferred(setindex(v, -2, 2)) === @SVector [1, -2, 3]

    # issue https://github.com/JuliaArrays/StaticArrays.jl/issues/1003
    @test @inferred(insert(SVector{0,Int}(), 1, 10)) === @SVector [10]

    @test_throws BoundsError insert(v, -2, 2)
    @test_throws BoundsError deleteat(v, -2)
    @test_throws BoundsError setindex(v, 2, -2)
end

@testset "setindex!" begin
    a = @MArray ones(3, 3, 3)
    a[2, 1, 3] = 2.
    @test a[2, 1, 3] == 2.
end

@testset "setindex" begin
    v1 = @SVector [1., 2., 3.]
    @test @inferred(setindex(v1, 5, 2)) == @SVector [1., 5., 3.]
    @test @inferred(setindex(v1, 5.0, 2)) == @SVector [1., 5., 3.]
    @test_throws BoundsError setindex(v1, 5.0, 0)
    @test_throws BoundsError setindex(v1, 5.0, 4)
    @test @inferred(setindex(v1, 5, CartesianIndex(2))) == setindex(v1, 5, 2)
    @test @inferred(setindex(v1, 5.0, CartesianIndex(2))) == setindex(v1, 5.0, 2)
    @test_throws BoundsError setindex(v1, 5.0, CartesianIndex(0))
    @test_throws BoundsError setindex(v1, 5.0, CartesianIndex(4))

    v2 = @SMatrix [1 2; 3 4]
    @test @inferred(setindex(v2, 7, 1)) == @SMatrix [7 2; 3 4]
    @test @inferred(setindex(v2, 7, 1, 1)) == @SMatrix [7 2; 3 4]
    @test @inferred(setindex(v2, 7, 2)) == @SMatrix [1 2; 7 4]
    @test @inferred(setindex(v2, 7, 2, 1)) == @SMatrix [1 2; 7 4]
    @test @inferred(setindex(v2, 7, 3)) == @SMatrix [1 7; 3 4]
    @test @inferred(setindex(v2, 7, 1, 2)) == @SMatrix [1 7; 3 4]
    @test @inferred(setindex(v2, 7, 4)) == @SMatrix [1 2; 3 7]
    @test @inferred(setindex(v2, 7, 2, 2)) == @SMatrix [1 2; 3 7]
    @test_throws BoundsError setindex(v2, 7, 0)
    @test_throws BoundsError setindex(v2, 7, 5)
    @test @inferred(setindex(v2, 7, CartesianIndex(1, 1))) == setindex(v2, 7, 1, 1)
    @test @inferred(setindex(v2, 7, CartesianIndex(2, 1))) == setindex(v2, 7, 2, 1)
    @test @inferred(setindex(v2, 7, CartesianIndex(1, 2))) == setindex(v2, 7, 1, 2)
    @test @inferred(setindex(v2, 7, CartesianIndex(2, 2))) == setindex(v2, 7, 2, 2)

    v3 = @SArray ones(2, 2, 2)
    @test @inferred(setindex(v3, 7, 2, 1, 2)) == reshape([1, 1, 1, 1, 1, 7, 1, 1], (2, 2, 2))
    @test_throws BoundsError setindex(v3, 7, 0)
    @test_throws BoundsError setindex(v3, 7, 9)
    @test @inferred(setindex(v3, 7, CartesianIndex(2, 1, 2))) == setindex(v3, 7, 2, 1, 2)

    # TODO: still missing proper multidimensional bounds checking
    # These should throw BoundsError, but don't
    @test_broken setindex(v3, 7, 3, 1, 2)
    @test_broken setindex(v3, 7, 2, 1, 0)
end





