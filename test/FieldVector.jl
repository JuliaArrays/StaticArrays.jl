@testset "FieldVector" begin
    @testset "Immutable Point3D" begin
        eval(quote
            struct Point3D <: FieldVector{3, Float64}
                x::Float64
                y::Float64
                z::Float64
            end

            # No need to define similar_type for non-parametric FieldVector (#792)
        end)

        p = Point3D(1.0, 2.0, 3.0)

        @test_throws Exception p[2] = 4.0

        @test size(p) === (3,)
        @test length(p) === 3
        @test eltype(p) === Float64

        @testinf Tuple(p) === (1.0, 2.0, 3.0)

        @test (p + p) === Point3D(2.0, 4.0, 6.0)

        @test (p[1], p[2], p[3]) === (p.x, p.y, p.z)

        m = @SMatrix [2.0 0.0 0.0;
                      0.0 2.0 0.0;
                      0.0 0.0 2.0]

        @test @inferred(m*p) === Point3D(2.0, 4.0, 6.0)
        @test @inferred(SA[2.0 0.0 0.0;
                           0.0 2.0 0.0]*p) === SVector((2.0, 4.0))

        @test @inferred(similar_type(Point3D)) == Point3D
        @test @inferred(similar_type(Point3D, Float64)) == Point3D
        @test @inferred(similar_type(Point3D, Float32)) == SVector{3,Float32}
        @test @inferred(similar_type(Point3D, Size(4))) == SVector{4,Float64}
        @test @inferred(similar_type(Point3D, Float32, Size(4))) == SVector{4,Float32}

        # Issue 146
        @test [[Point3D(1.0,2.0,3.0)]; [Point3D(4.0,5.0,6.0)]]::Vector{Point3D} == [Point3D(1.0,2.0,3.0), Point3D(4.0,5.0,6.0)]

        # Issue 342
        @test_throws DimensionMismatch("No precise constructor for Point3D found. Length of input was 4.") Point3D(1,2,3,4)

        eval(quote
            struct Point2DT{T} <: FieldVector{2, T}
                x::T
                y::T
            end
        end)

        # eltype promotion
        @test Point2DT(1., 2) === Point2DT(1.0, 2.0) && Tuple(Point2DT(1.0, 2.0)) === (1.0, 2.0)
        @test Point2DT{Int}(1., 2) === Point2DT(1, 2) && Tuple(Point2DT(1, 2)) === (1, 2)
    end

    @testset "Mutable Point2D" begin
        eval(quote
            mutable struct Point2D{T} <: FieldVector{2, T}
                x::T
                y::T
            end

            StaticArrays.similar_type(::Type{<:Point2D}, ::Type{T}, s::Size{(2,)}) where {T} = Point2D{T}
        end)

        p = Point2D(0.0, 0.0)
        p[1] = 1.0
        p[2] = 2.0

        @test size(p) === (2,)
        @test length(p) === 2
        @test eltype(p) === Float64

        @testinf Tuple(p) === (1.0, 2.0)

        @test (p[1], p[2]) === (p.x, p.y)
        @test (p[1], p[2]) === (1.0, 2.0)

        m = @SMatrix [2.0 0.0;
                      0.0 2.0]

        @test @inferred(m*p)::Point2D == Point2D(2.0, 4.0)

        @test @inferred(similar_type(Point2D{Float64})) == Point2D{Float64}
        @test @inferred(similar_type(Point2D{Float64}, Float32)) == Point2D{Float32}
        @test @inferred(similar_type(Point2D{Float64}, Size(4))) == SVector{4,Float64}
        @test @inferred(similar_type(Point2D{Float64}, Float32, Size(4))) == SVector{4,Float32}

        # eltype promotion
        @test Point2D(1f0, 2) isa Point2D{Float32}
        @test Point2D{Int}(1f0, 2) isa Point2D{Int}
        p = Point2D(0, 0.0)
        @test p[1] === p[2] === 0.0
    end

    @testset "FieldVector with Tuple fields" begin
        # verify that having a field which is itself a Tuple
        # doesn't break anything

        eval(quote
            struct TupleField <: FieldVector{1, NTuple{2, Int}}
                x::NTuple{2, Int}
                function TupleField(x::NTuple{2,Int})
                    new(x)
                end
            end
        end)

        x = TupleField((1,2))
        @test length(x) == 1
        @test length(x[1]) == 2
        @test x.x == (1, 2)
    end

    @testset "FieldVector with parametric eltype and without similar_type" begin
        eval(quote
            struct FVT{T} <: FieldVector{2, T}
                x::T
                y::T
            end

            # No similar_type defined - test fallback codepath
        end)

        @test @inferred(similar_type(FVT{Float64}, Float32)) == SVector{2,Float32} # Fallback code path
        @test @inferred(similar_type(FVT{Float64}, Size(2))) == FVT{Float64}
        @test @inferred(similar_type(FVT{Float64}, Size(3))) == SVector{3,Float64}
        @test @inferred(similar_type(FVT{Float64}, Float32, Size(3))) == SVector{3,Float32}
    end

    @testset "FieldVector with constructor missing" begin
        struct Position1088{T} <: FieldVector{3, T}
            x::T
            y::T
            z::T
            Position1088(x::T, y::T, z::T) where {T} = new{T}(x, y, z)
        end
        @test_throws ErrorException("The constructor for Position1088{Float64}(::Float64, ::Float64, ::Float64) is missing!") Position1088((1.,2.,3.))
    end

    @testset "FieldVector to NamedTuple" begin
        struct FieldVectorNT{T} <: FieldVector{3,T}
            a::T
            b::T
            c::T
        end

        @test NamedTuple(FieldVectorNT(1,2,3)) isa @NamedTuple{a::Int, b::Int, c::Int}
    end
end
