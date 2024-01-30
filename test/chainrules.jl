using StaticArrays, ChainRulesCore, ChainRulesTestUtils, JLArrays, LinearAlgebra, Test

@testset "ChainRules Integration" begin
    @testset "Projection" begin
        # There is no code for this, but when argument isa StaticArray, axes(x) === axes(dx)
        # implies a check, and reshape will wrap a Vector into a static SizedVector:
        pstat = ProjectTo(SA[1, 2, 3])
        @test axes(pstat(rand(3))) === (SOneTo(3),)

        # This recurses into structured arrays:
        pst = ProjectTo(transpose(SA[1, 2, 3]))
        @test axes(pst(rand(1,3))) === (SOneTo(1), SOneTo(3))
        @test pst(rand(1,3)) isa Transpose

        # When the argument is an ordinary Array, static gradients are allowed to pass,
        # like FillArrays. Collecting to an Array would cost a copy.
        pvec3 = ProjectTo([1, 2, 3])
        @test pvec3(SA[1, 2, 3]) isa StaticArray
    end

    @testset "Constructor rrules" begin
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

    @testset "Array of Structs Projection" begin
        x = JLArray(rand(SVector{3, Float64}, 10))
        @inferred ProjectTo(x)
        @inferred Union{Nothing, JLVector{SVector{3, Float64}}, DenseJLVector{SVector{3, Float64}}} ProjectTo(x)(x)
        @test ProjectTo(x)(x) isa JLArray
    end
end
