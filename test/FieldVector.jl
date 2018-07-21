@testset "FieldVector" begin
    @testset "Immutable Point3D" begin
        eval(quote
            struct Point3D <: FieldVector{3, Float64}
                x::Float64
                y::Float64
                z::Float64
            end

            StaticArrays.similar_type(::Type{Point3D}, ::Type{Float64}, s::Size{(3,)}) = Point3D
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

        @test @inferred(similar_type(Point3D)) == Point3D
        @test @inferred(similar_type(Point3D, Float64)) == Point3D
        @test @inferred(similar_type(Point3D, Float32)) == SVector{3,Float32}
        @test @inferred(similar_type(Point3D, Size(4))) == SVector{4,Float64}
        @test @inferred(similar_type(Point3D, Float32, Size(4))) == SVector{4,Float32}

        # Issue 146
        @test [[Point3D(1.0,2.0,3.0)]; [Point3D(4.0,5.0,6.0)]]::Vector{Point3D} == [Point3D(1.0,2.0,3.0), Point3D(4.0,5.0,6.0)]

        # Issue 342
        @test_throws ErrorException Point3D(1,2,3,4)
    end

    @testset "Mutable Point2D" begin
        eval(quote
            mutable struct Point2D{T} <: FieldVector{2, T}
                x::T
                y::T
            end

            StaticArrays.similar_type(::Type{P2D}, ::Type{T}, s::Size{(2,)}) where {P2D<:Point2D,T} = Point2D{T}
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
    end

    @testset "FieldVector with Tuple fields" begin
        # verify that having a field which is itself a Tuple 
        # doesn't break anything

        eval(quote
            struct TupleField <: FieldVector{1, NTuple{2, Int}}
                x::NTuple{2, Int}
            end
        end)

        x = TupleField((1,2))
        @test length(x) == 1
        @test length(x[1]) == 2
        @test x.x == (1, 2)
    end
end
