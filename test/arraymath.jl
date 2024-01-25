using StaticArrays, Test
import StaticArrays.arithmetic_closure

struct TestDie
    nsides::Int
end
Random.rand(rng::AbstractRNG, d::Random.SamplerTrivial{TestDie}) = rand(rng, 1:d[].nsides)
Base.eltype(::Type{TestDie}) = Int

const AbstractFloatSubtypes = [Float16, Float32, Float64, BigFloat]
const UnsignedSubtypes = [UInt8, UInt16, UInt32, UInt64, UInt128]
const SignedSubtypes = [Int8, Int16, Int32, Int64, Int128, BigInt]

@testset "Array math" begin
    @testset "zeros() and ones()" begin
        @test @inferred(zeros(SVector{3,Float64})) === @SVector [0.0, 0.0, 0.0]
        @test @inferred(zeros(SVector{3,Int})) === @SVector [0, 0, 0]
        @test @inferred(ones(SVector{3,Float64})) === @SVector [1.0, 1.0, 1.0]
        @test @inferred(ones(SVector{3,Int})) === @SVector [1, 1, 1]
        @test @inferred(zeros(SVector{0,Float64})) === @SVector Float64[]
        @test @inferred(zeros(SVector{0,Int})) === @SVector Int[]
        @test @inferred(ones(SVector{0,Float64})) === @SVector Float64[]
        @test @inferred(ones(SVector{0,Int})) === @SVector Int[]

        @test @inferred(zeros(SVector{3})) === @SVector [0.0, 0.0, 0.0]
        @test @inferred(zeros(SMatrix{2,2})) === @SMatrix [0.0 0.0; 0.0 0.0]
        @test @inferred(zeros(SArray{Tuple{1,1,1}})) === SArray{Tuple{1,1,1}}((0.0,))
        @test @inferred(zeros(MVector{3}))::MVector == @MVector [0.0, 0.0, 0.0]
        @test @inferred(zeros(MMatrix{2,2}))::MMatrix == @MMatrix [0.0 0.0; 0.0 0.0]
        @test @inferred(zeros(MArray{Tuple{1,1,1}}))::MArray == MArray{Tuple{1,1,1}}((0.0,))

        @test @inferred(ones(SVector{3})) === @SVector [1.0, 1.0, 1.0]
        @test @inferred(ones(SMatrix{2,2})) === @SMatrix [1.0 1.0; 1.0 1.0]
        @test @inferred(ones(SArray{Tuple{1,1,1}})) === SArray{Tuple{1,1,1}}((1.0,))
        @test @inferred(ones(MVector{3}))::MVector == @MVector [1.0, 1.0, 1.0]
        @test @inferred(ones(MMatrix{2,2}))::MMatrix == @MMatrix [1.0 1.0; 1.0 1.0]
        @test @inferred(ones(MArray{Tuple{1,1,1}}))::MArray == MArray{Tuple{1,1,1}}((1.0,))

        # https://github.com/JuliaArrays/StaticArrays.jl/issues/428
        bigzeros = zeros(SVector{2, BigInt})
        @test bigzeros == @SVector [big(0), big(0)]
        @test bigzeros[1] !== bigzeros[2]

        bigones = ones(SVector{2, BigInt})
        @test bigones == @SVector [big(1), big(1)]
        @test bigones[1] !== bigones[2]
    end

    @testset "ones()" begin
        for SA in (SVector, MVector, SizedVector)
            # Float64
            m = @inferred ones(SA{3, Float64})
            @test m == [1.0, 1.0, 1.0]
            @test m isa SA{3, Float64}
            # Int
            m = @inferred ones(SA{3, Int})
            @test m == [1, 1, 1]
            @test m isa SA{3, Int}
            # Unspecified
            m = @inferred ones(SA{3})
            @test m == [1.0, 1.0, 1.0]
            @test m isa SA{3}
            # Float64
            m = @inferred ones(SA{0, Float64})
            @test m == Float64[]
            @test m isa SA{0, Float64}
            # Int
            m = @inferred ones(SA{0, Int})
            @test m == Int[]
            @test m isa SA{0, Int}
            # Unspecified
            m = @inferred ones(SA{0})
            @test m == Float64[]
            @test m isa SA{0}
            # Any
            @test_throws MethodError ones(SA{3, Any})
            @test ones(SA{0, Any}) isa SA{0, Any}
        end
    end

    @testset "zero()" begin
        for SA in (SVector, MVector, SizedVector)
            # Float64
            m = @inferred zero(SA{3, Float64})
            @test m == [0.0, 0.0, 0.0]
            @test m isa SA{3, Float64}
            # Int
            m = @inferred zero(SA{3, Int})
            @test m == [0, 0, 0]
            @test m isa SA{3, Int}
            # Unspecified
            m = @inferred zero(SA{3})
            @test m == [0.0, 0.0, 0.0]
            @test m isa SA{3}
            # Float64 (zero-element)
            m = @inferred zero(SA{0, Float64})
            @test m == Float64[]
            @test m isa SA{0, Float64}
            # Int (zero-element)
            m = @inferred zero(SA{0, Int})
            @test m == Int[]
            @test m isa SA{0, Int}
            # Unspecified (zero-element)
            m = @inferred zero(SA{0})
            @test m == Float64[]
            @test m isa SA{0}
            # Any
            @test_throws MethodError zeros(SA{3, Any})
            @test zeros(SA{0, Any}) isa SA{0, Any}
        end
    end

    @testset "fill()" begin
        @test @allocated(fill(0., SMatrix{1, 16, Float64})) == 0 # #81
        @test @allocated(fill(0., SMatrix{0, 5, Float64})) == 0

        for SA in (SMatrix, MMatrix, SizedMatrix)
            for T in (Float64, Int, Any)
                # Float64 -> T
                m = @inferred(fill(3.0, SA{4, 16, T}))
                @test m isa SA{4, 16, T}
                @test all(m .== 3)
                # Float64 -> T (zero-element)
                m = @inferred(fill(3.0, SA{0, 5, T}))
                @test m isa SA{0, 5, T}
                @test all(m .== 3)
                # Int -> T
                m = @inferred(fill(3, SA{4, 16, T}))
                @test m isa SA{4, 16, T}
                @test all(m .== 3)
                # Int -> T (zero-element)
                m = @inferred(fill(3, SA{0, 5, T}))
                @test m isa SA{0, 5, T}
                @test all(m .== 3)
            end

            # Float64 -> Unspecified
            m = @inferred(fill(3.0, SA{4, 16}))
            @test m isa SA{4, 16, Float64}
            @test all(m .== 3)
            # Float64 -> Unspecified (zero-element)
            m = @inferred(fill(3.0, SA{0, 5}))
            @test m isa SA{0, 5, Float64}
            @test all(m .== 3)
            # Int -> Unspecified
            m = @inferred(fill(3, SA{4, 16}))
            @test m isa SA{4, 16, Int}
            @test all(m .== 3)
            # Int -> Unspecified (zero-element)
            m = @inferred(fill(3, SA{0, 5}))
            @test m isa SA{0, 5, Int}
            @test all(m .== 3)
        end
    end

    @testset "fill!()" begin
        m = MMatrix{4,16,Float64}(undef)
        fill!(m, 3)
        @test all(m .== 3.)
        m = MMatrix{0,5,Float64}(undef)
        fill!(m, 3)
        @test all(m .== 3.)
    end

    @testset "rand()" begin
        m = rand(1:2, SVector{3})
        check = ((m .>= 1) .& (m .<= 2))
        @test all(check)
        m = rand(1:2, SMatrix{4, 4})
        check = ((m .>= 1) .& (m .<= 2))
        @test all(check)
        m = rand(1:1, SVector{3})
        @test rand(m) == 1

        for SA in (SVector, MVector, SizedVector)
            v1 = rand(SA{3})
            @test v1 isa SA{3, Float64}
            @test all(0 .< v1 .< 1)

            v2 = rand(SA{0})
            @test v2 isa SA{0, Float64}
            @test all(0 .< v2 .< 1)

            v3 = rand(SA{3, Float32})
            @test v3 isa SA{3, Float32}
            @test all(0 .< v3 .< 1)

            v4 = rand(SA{0, Float32})
            @test v4 isa SA{0, Float32}
            @test all(0 .< v4 .< 1)
        end
        rng = MersenneTwister(123)
        @test (@SVector rand(3)) isa SVector{3,Float64}
        @test (@SMatrix rand(3, 4)) isa SMatrix{3,4,Float64}
        @test (@SArray rand(3, 4, 5)) isa SArray{Tuple{3,4,5},Float64}

        @test (@MVector rand(3)) isa MVector{3,Float64}
        @test (@MMatrix rand(3, 4)) isa MMatrix{3,4,Float64}
        @test (@MArray rand(3, 4, 5)) isa MArray{Tuple{3,4,5},Float64}

        @test (@SVector rand(TestDie(6), 3)) isa SVector{3,Int}
        @test (@SVector rand(rng, TestDie(6), 3)) isa SVector{3,Int}
        @test (@SVector rand(TestDie(6), 0)) isa SVector{0,Int}
        @test (@SVector rand(rng, TestDie(6), 0)) isa SVector{0,Int}
        @test (@MVector rand(TestDie(6), 3)) isa MVector{3,Int}
        @test (@MVector rand(rng, TestDie(6), 3)) isa MVector{3,Int}

        @test (@SMatrix rand(TestDie(6), 3, 4)) isa SMatrix{3,4,Int}
        @test (@SMatrix rand(rng, TestDie(6), 3, 4)) isa SMatrix{3,4,Int}
        @test (@SMatrix rand(TestDie(6), 0, 4)) isa SMatrix{0,4,Int}
        @test (@SMatrix rand(rng, TestDie(6), 0, 4)) isa SMatrix{0,4,Int}
        @test (@MMatrix rand(TestDie(6), 3, 4)) isa MMatrix{3,4,Int}
        @test (@MMatrix rand(rng, TestDie(6), 3, 4)) isa MMatrix{3,4,Int}

        @test (@SArray rand(TestDie(6), 3, 4, 5)) isa SArray{Tuple{3,4,5},Int}
        @test (@SArray rand(rng, TestDie(6), 3, 4, 5)) isa SArray{Tuple{3,4,5},Int}
        @test (@SArray rand(TestDie(6), 0, 4, 5)) isa SArray{Tuple{0,4,5},Int}
        @test (@SArray rand(rng, TestDie(6), 0, 4, 5)) isa SArray{Tuple{0,4,5},Int}
        @test (@MArray rand(TestDie(6), 3, 4, 5)) isa MArray{Tuple{3,4,5},Int}

        # test if rng generator is actually respected
        @test (@SVector rand(MersenneTwister(123), TestDie(6), 3)) ===
              (@SVector rand(MersenneTwister(123), TestDie(6), 3))
        @test (@SMatrix rand(MersenneTwister(123), TestDie(6), 3, 4)) ===
              (@SMatrix rand(MersenneTwister(123), TestDie(6), 3, 4))
        @test (@SArray rand(MersenneTwister(123), TestDie(6), 3, 4, 5)) ===
              (@SArray rand(MersenneTwister(123), TestDie(6), 3, 4, 5))
    end

    @testset "rand!()" begin
        m = @MMatrix [0. 0.; 0. 0.]
        rand!(m)
        check = ((m .< 1.) .& (m .> 0.))
        @test all(check)
        m = @MMatrix [0. 0.; 0. 0.]
        rand!(m, 1:2)
        check = ((m .>= 1) .& (m .<= 2))
        @test all(check)

        for SA in (MVector, SizedVector)
            v1 = rand(SA{3})
            rand!(v1)
            @test v1 isa SA{3, Float64}
            @test all(0 .< v1 .< 1)

            v2 = rand(SA{0})
            rand!(v2)
            @test v2 isa SA{0, Float64}
            @test all(0 .< v2 .< 1)

            v3 = rand(SA{3, Float32})
            rand!(v3)
            @test v3 isa SA{3, Float32}
            @test all(0 .< v3 .< 1)

            v4 = rand(SA{0, Float32})
            rand!(v4)
            @test v4 isa SA{0, Float32}
            @test all(0 .< v4 .< 1)
        end
    end

    @testset "randn()" begin
        for SA in (SVector, MVector, SizedVector)
            v1 = randn(SA{3})
            @test v1 isa SA{3, Float64}

            v2 = randn(SA{0})
            @test v2 isa SA{0, Float64}

            v3 = randn(SA{3, Float32})
            @test v3 isa SA{3, Float32}

            v4 = randn(SA{0, Float32})
            @test v4 isa SA{0, Float32}
        end
    end

    @testset "randn!()" begin
        for SA in (MVector, SizedVector)
            v1 = randn(SA{3})
            randn!(v1)
            @test v1 isa SA{3, Float64}

            v2 = randn(SA{0})
            randn!(v2)
            @test v2 isa SA{0, Float64}

            v3 = randn(SA{3, Float32})
            randn!(v3)
            @test v3 isa SA{3, Float32}

            v4 = randn(SA{0, Float32})
            randn!(v4)
            @test v4 isa SA{0, Float32}
        end
    end

    @testset "randexp()" begin
        for SA in (SVector, MVector, SizedVector)
            v1 = randexp(SA{3})
            @test v1 isa SA{3, Float64}
            @test all(0 .< v1)

            v2 = randexp(SA{0})
            @test v2 isa SA{0, Float64}
            @test all(0 .< v2)

            v3 = randexp(SA{3, Float32})
            @test v3 isa SA{3, Float32}
            @test all(0 .< v3)

            v4 = randexp(SA{0, Float32})
            @test v4 isa SA{0, Float32}
            @test all(0 .< v4)
        end
    end

    @testset "randexp!()" begin
        for SA in (MVector, SizedVector)
            v1 = randexp(SA{3})
            randexp!(v1)
            @test v1 isa SA{3, Float64}
            @test all(0 .< v1)

            v2 = randexp(SA{0})
            randexp!(v2)
            @test v2 isa SA{0, Float64}
            @test all(0 .< v2)

            v3 = randexp(SA{3, Float32})
            randexp!(v3)
            @test v3 isa SA{3, Float32}
            @test all(0 .< v3)

            v4 = randexp(SA{0, Float32})
            randexp!(v4)
            @test v4 isa SA{0, Float32}
            @test all(0 .< v4)
        end
    end

    @testset "arithmetic_closure" for T0 in [UnsignedSubtypes;
                                             SignedSubtypes;
                                             AbstractFloatSubtypes;
                                             Bool;
                                             Complex{Int};
                                             Complex{Float64};
                                             ]
        T = @inferred arithmetic_closure(T0)
        @test arithmetic_closure(T) == T

        t = one(T)
        @test (t+t) isa T
        @test (t-t) isa T
        @test (t*t) isa T
        @test (t/t) isa T
    end

    @testset "arithmetic_closure allocation" begin
        # a little icky, but `@allocated` seems to be too fragile to use in a loop with
        # types assigned in variables (see #924); so we write out a test explicitly for
        # every `isbitstype` type of interest
        @test (@allocated arithmetic_closure(UInt128))        == 0
        @test (@allocated arithmetic_closure(UInt16))         == 0
        @test (@allocated arithmetic_closure(UInt32))         == 0
        @test (@allocated arithmetic_closure(UInt64))         == 0
        @test (@allocated arithmetic_closure(UInt8))          == 0
        @test (@allocated arithmetic_closure(Int128))         == 0
        @test (@allocated arithmetic_closure(Int16))          == 0
        @test (@allocated arithmetic_closure(Int32))          == 0
        @test (@allocated arithmetic_closure(Int64))          == 0
        @test (@allocated arithmetic_closure(Int8))           == 0
        @test (@allocated arithmetic_closure(Float16))        == 0
        @test (@allocated arithmetic_closure(Float32))        == 0
        @test (@allocated arithmetic_closure(Float64))        == 0
        @test (@allocated arithmetic_closure(Bool))           == 0
        @test (@allocated arithmetic_closure(Complex{Int64})) == 0
        @test (@allocated arithmetic_closure(ComplexF64))     == 0
    end
end
