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
end
