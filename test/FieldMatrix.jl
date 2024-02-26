@testset "FieldMatrix" begin
    @testset "Immutable Tensor3x3" begin
        eval(quote
            struct Tensor3x3 <: FieldMatrix{3, 3, Float64}
                xx::Float64
                yx::Float64
                zx::Float64
                xy::Float64
                yy::Float64
                zy::Float64
                xz::Float64
                yz::Float64
                zz::Float64
            end

            # No need to define similar_type for non-parametric FieldMatrix (#792)
        end)

        p = Tensor3x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

        @test_throws Exception p[2] = 4.0
        @test_throws Exception p[2, 3] = 6.0

        @test size(p) === (3,3)
        @test length(p) === 9
        @test eltype(p) === Float64

        @testinf Tuple(p) === (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

         @test @inferred(p + p) === Tensor3x3(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)

        @test (p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]) === (p.xx, p.yx, p.zx, p.xy, p.yy, p.zy, p.xz, p.yz, p.zz)

        m = @SMatrix [2.0 0.0 0.0;
                      0.0 2.0 0.0;
                      0.0 0.0 2.0]

        @test @inferred(p*m) === Tensor3x3(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)

        @test @inferred(similar_type(Tensor3x3)) == Tensor3x3
        @test @inferred(similar_type(Tensor3x3, Float64)) == Tensor3x3
        @test @inferred(similar_type(Tensor3x3, Float32)) == SMatrix{3,3,Float32,9}
        @test @inferred(similar_type(Tensor3x3, Size(4,4))) == SMatrix{4,4,Float64,16}
        @test @inferred(similar_type(Tensor3x3, Float32, Size(4,4))) == SMatrix{4,4,Float32,16}

        # Issue 146
        @test [[Tensor3x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)]; [Tensor3x3(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)]]::Vector{Tensor3x3} == [Tensor3x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), Tensor3x3(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)]

        # Issue 342
        @test_throws DimensionMismatch("No precise constructor for Tensor3x3 found. Length of input was 10.") Tensor3x3(1,2,3,4,5,6,7,8,9,10)
    end

    @testset "Mutable Tensor2x2" begin
        eval(quote
            mutable struct Tensor2x2{T} <: FieldMatrix{2, 2, T}
                xx::T
                yx::T
                xy::T
                yy::T
            end

            StaticArrays.similar_type(::Type{<:Tensor2x2}, ::Type{T}, s::Size{(2,2)}) where {T} = Tensor2x2{T}
        end)

        p = Tensor2x2(0.0, 0.0, 0.0, 0.0)
        p[1,1] = 1.0
        @test setindex!(p, 1.0, 1,1) === p
        p[2,1] = 2.0

        @test size(p) === (2,2)
        @test length(p) === 4
        @test eltype(p) === Float64

        @testinf Tuple(p) === (1.0, 2.0, 0.0, 0.0)

        @test (p[1], p[2], p[3], p[4]) === (p.xx, p.yx, p.xy, p.yy)
        @test (p[1], p[2], p[3], p[4]) === (1.0, 2.0, 0.0, 0.0)

        m = @SMatrix [2.0 0.0;
                      0.0 2.0]

        @test @inferred(p*m)::Tensor2x2 == Tensor2x2(2.0, 4.0, 0.0, 0.0)

        @test @inferred(similar_type(Tensor2x2{Float64})) == Tensor2x2{Float64}
        @test @inferred(similar_type(Tensor2x2{Float64}, Float32)) == Tensor2x2{Float32}
        @test @inferred(similar_type(Tensor2x2{Float64}, Size(3,3))) == SMatrix{3,3,Float64,9}
        @test @inferred(similar_type(Tensor2x2{Float64}, Float32, Size(4,4))) == SMatrix{4,4,Float32,16}

        # eltype promotion
        @test Tuple(@inferred(Tensor2x2(1., 2, 3, 4f0))) === (1.,2.,3.,4.)
        @test Tuple(@inferred(Tensor2x2{Int}(1., 2, 3, 4f0))) === (1,2,3,4)
    end

    @testset "FieldMatrix with Tuple fields" begin
        # verify that having a field which is itself a Tuple
        # doesn't break anything

        eval(quote
            struct TupleField2 <: FieldMatrix{1, 1, NTuple{2, Int}}
                x::NTuple{2, Int}
                function TupleField2(x::NTuple{2,Int})
                    new(x)
                end
            end
        end)

        x = TupleField2((1,2))
        @test length(x) == 1
        @test length(x[1]) == 2
        @test x.x == (1, 2)
    end

    @testset "FieldMatrix to NamedTuple" begin
        struct FieldMatrixNT{T} <: FieldMatrix{2,2,T}
            a::T
            b::T
            c::T
            d::T
        end

        @test NamedTuple(FieldMatrixNT(1,2,3,4)) isa @NamedTuple{a::Int, b::Int, c::Int, d::Int}
    end
end
