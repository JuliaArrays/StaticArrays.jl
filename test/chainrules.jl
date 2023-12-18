using StaticArrays, ChainRulesCore, ChainRulesTestUtils, Test

@testset "Chain Rules Integration" begin
    @testset "Projection" begin
        test_rrule(SMatrix{1, 4}, (1.0, 1.0, 1.0, 1.0))
        test_rrule(SMatrix{4, 1}, (1.0, 1.0, 1.0, 1.0))
        test_rrule(SMatrix{2, 2}, (1.0, 1.0, 1.0, 1.0))
        test_rrule(SVector{4}, (1.0, 1.0, 1.0, 1.0))
        test_rrule(SVector{4}, 1.0, 1.0, 1.0, 1.0)
        test_rrule(SVector{4}, 1.0, 1.0f0, 1.0, 1.0f0)
    end

    @testset "Type Stability" begin
        x = ones(SMatrix{2, 2})
        y = ones(SVector{4})

        @inferred ProjectTo(x)
        @inferred ProjectTo(y)
        @inferred ProjectTo(x)(y)
        @inferred ProjectTo(y)(x)

        x = ones(SMatrix{2, 2, Float32})
        y = ones(SVector{4})

        @inferred ProjectTo(x)
        @inferred ProjectTo(x)(y)
        @inferred ProjectTo(y)(x)
    end
end
