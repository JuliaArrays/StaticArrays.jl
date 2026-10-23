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

    @testset "ccall" begin
        eval(quote
            struct CPoint3D <: FieldVector{3, Float64}
                x::Float64
                y::Float64
                z::Float64
            end
            mutable struct MCPoint3D <: FieldVector{3, Float64}
                x::Float64
                y::Float64
                z::Float64
            end
        end)

        # `memmove` reads the elements through the source pointer and writes
        # them through the destination pointer.
        load_elements(x) = (y = zeros(3); ccall(:memmove, Ptr{Cvoid}, (Ptr{Float64}, Ptr{Float64}, Csize_t), y, x, 24); y)
        load_void(x) = (y = zeros(3); ccall(:memmove, Ptr{Cvoid}, (Ptr{Float64}, Ptr{Cvoid}, Csize_t), y, x, 24); y)
        load_array(x::X) where {X} = (y = zeros(3); ccall(:memmove, Ptr{Cvoid}, (Ptr{Float64}, Ptr{X}, Csize_t), y, x, 24); y)
        store_elements(r) = ccall(:memmove, Ptr{Cvoid}, (Ptr{Float64}, Ptr{Float64}, Csize_t), r, [4.0, 5.0, 6.0], 24)
        store_void(r) = ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Float64}, Csize_t), r, [4.0, 5.0, 6.0], 24)
        store_array(r::Base.RefValue{X}) where {X} = ccall(:memmove, Ptr{Cvoid}, (Ptr{X}, Ptr{Float64}, Csize_t), r, [4.0, 5.0, 6.0], 24)
        load_ref_array(r::Base.RefValue{X}) where {X} = (y = zeros(3); ccall(:memmove, Ptr{Cvoid}, (Ptr{Float64}, Ptr{X}, Csize_t), y, r, 24); y)

        for P in (CPoint3D, MCPoint3D)
            p = P(1.0, 2.0, 3.0)
            for f in (load_elements, load_void, load_array)
                @test f(p) == [1.0, 2.0, 3.0]
            end
            for f in (load_elements, load_void)
                @test f(Ref(p)) == [1.0, 2.0, 3.0]
            end
            # A `Ref` with an abstract element type holds a boxed array, which Base
            # passes as the address of the box's elements.
            r = Ref{FieldVector{3, Float64}}(p)
            @test load_void(r) == [1.0, 2.0, 3.0]
            @test load_ref_array(r) == [1.0, 2.0, 3.0]
            for f in (store_elements, store_void, store_array)
                r = Ref(P(1.0, 2.0, 3.0))
                f(r)
                @test r[] == [4.0, 5.0, 6.0]
            end
        end
        # A mutable FieldVector is passed in place.
        m = MCPoint3D(1.0, 2.0, 3.0)
        ccall(:memmove, Ptr{Cvoid}, (Ptr{Float64}, Ptr{Float64}, Csize_t), m, [4.0, 5.0, 6.0], 24)
        @test m == [4.0, 5.0, 6.0]
    end
end
