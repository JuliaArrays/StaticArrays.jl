using StaticArrays, Test

@testset "accumulate" begin
    @testset "cumsum(::$label)" for (label, T) in [
        # label, T
        ("SVector", SVector),
        ("MVector", MVector),
        ("SizedVector", SizedVector),
    ]
        @testset "$label" for (label, a) in [
            ("[1, 2, 3]", T{3}(SA[1, 2, 3])),
            ("[]", T{0,Int}(())),
        ]
            @test cumsum(a) == cumsum(collect(a))
            @test cumsum(a) isa similar_type(a)
            @inferred cumsum(a)
        end
        @test eltype(cumsum(T{0,Int8}(()))) == eltype(cumsum(Int8[]))
        @test eltype(cumsum(T{1,Int8}((1)))) == eltype(cumsum(Int8[1]))
        @test eltype(cumsum(T{2,Int8}((1, 2)))) == eltype(cumsum(Int8[1, 2]))
    end

    @testset "cumsum(::$label; dims=2)" for (label, T) in [
        # label, T
        ("SMatrix", SMatrix),
        ("MMatrix", MMatrix),
        ("SizedMatrix", SizedMatrix),
    ]
        @testset "$label" for (label, a) in [
            ("[1 2; 3 4; 5 6]", T{3,2}(SA[1 2; 3 4; 5 6])),
            ("0 x 2 matrix", T{0,2,Float64}()),
            ("2 x 0 matrix", T{2,0,Float64}()),
        ]
            @test cumsum(a; dims = 2) == cumsum(collect(a); dims = 2)
            @test cumsum(a; dims = 2) isa similar_type(a)
            @inferred cumsum(a; dims = Val(2))
        end
    end

    @testset "cumsum(a::SArray; dims=$i); ndims(a) = $d" for d in 1:4, i in 1:d
        shape = Tuple(1:d)
        a = similar_type(SArray, Int, Size(shape))(1:prod(shape))
        @test cumsum(a; dims = i) == cumsum(collect(a); dims = i)
        @test cumsum(a; dims = i) isa SArray
        @inferred cumsum(a; dims = Val(i))
    end

    @testset "cumprod" begin
        a = SA[1, 2, 3]
        @test cumprod(a)::SArray == cumprod(collect(a))
        @inferred cumprod(a)

        @test eltype(cumsum(SA{Int8}[])) == eltype(cumsum(Int8[]))
        @test eltype(cumsum(SA{Int8}[1])) == eltype(cumsum(Int8[1]))
        @test eltype(cumsum(SA{Int8}[1, 2])) == eltype(cumsum(Int8[1, 2]))
    end

    @testset "empty vector with init" begin
        a = SA{Int}[]
        right(_, x) = x
        @test accumulate(right, a; init = Val(1)) === SA{Int}[]
        @inferred accumulate(right, a; init = Val(1))
    end
end
