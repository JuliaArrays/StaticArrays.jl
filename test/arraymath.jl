using StaticArrays, Test
import StaticArrays.arithmetic_closure

@testset "Array math" begin
    @testset "zeros() and ones()" begin
        @test @inferred(zeros(SVector{3,Float64})) === @SVector [0.0, 0.0, 0.0]
        @test @inferred(zeros(SVector{3,Int})) === @SVector [0, 0, 0]
        @test @inferred(ones(SVector{3,Float64})) === @SVector [1.0, 1.0, 1.0]
        @test @inferred(ones(SVector{3,Int})) === @SVector [1, 1, 1]

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

    @testset "zero()" begin
        @test @inferred(zero(SVector{3, Float64})) === @SVector [0.0, 0.0, 0.0]
        @test @inferred(zero(SVector{3, Int})) === @SVector [0, 0, 0]
    end

    @testset "fill()" begin
        @test all(@inferred(fill(3., SMatrix{4, 16, Float64})) .== 3.)
        @test @allocated(fill(0., SMatrix{1, 16, Float64})) == 0 # #81
    end

    @testset "fill!()" begin
        m = MMatrix{4,16,Float64}(undef)
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

        for T in (SVector, MVector, SizedVector)
            v1 = rand(T{3})
            @test v1 isa T{3, Float64}
            @test all(0 .< v1 .< 1)

            v2 = rand(T{0})
            @test v2 isa T{0, Float64}
            @test all(0 .< v2 .< 1)

            v3 = rand(T{3, Float32})
            @test v3 isa T{3, Float32}
            @test all(0 .< v3 .< 1)

            v4 = rand(T{0, Float32})
            @test v4 isa T{0, Float32}
            @test all(0 .< v4 .< 1)
        end
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

        for T in (MVector, SizedVector)
            v1 = rand(T{3})
            rand!(v1)
            @test v1 isa T{3, Float64}
            @test all(0 .< v1 .< 1)

            v2 = rand(T{0})
            rand!(v2)
            @test v2 isa T{0, Float64}
            @test all(0 .< v2 .< 1)

            v3 = rand(T{3, Float32})
            rand!(v3)
            @test v3 isa T{3, Float32}
            @test all(0 .< v3 .< 1)

            v4 = rand(T{0, Float32})
            rand!(v4)
            @test v4 isa T{0, Float32}
            @test all(0 .< v4 .< 1)
        end
    end

    @testset "randn()" begin
        for T in (SVector, MVector, SizedVector)
            v1 = randn(T{3})
            @test v1 isa T{3, Float64}

            v2 = randn(T{0})
            @test v2 isa T{0, Float64}

            v3 = randn(T{3, Float32})
            @test v3 isa T{3, Float32}

            v4 = randn(T{0, Float32})
            @test v4 isa T{0, Float32}
        end
    end

    @testset "randn!()" begin
        for T in (MVector, SizedVector)
            v1 = randn(T{3})
            randn!(v1)
            @test v1 isa T{3, Float64}

            v2 = randn(T{0})
            randn!(v2)
            @test v2 isa T{0, Float64}

            v3 = randn(T{3, Float32})
            randn!(v3)
            @test v3 isa T{3, Float32}

            v4 = randn(T{0, Float32})
            randn!(v4)
            @test v4 isa T{0, Float32}
        end
    end

    @testset "randexp()" begin
        for T in (SVector, MVector, SizedVector)
            v1 = randexp(T{3})
            @test v1 isa T{3, Float64}
            @test all(0 .< v1)

            v2 = randexp(T{0})
            @test v2 isa T{0, Float64}
            @test all(0 .< v2)

            v3 = randexp(T{3, Float32})
            @test v3 isa T{3, Float32}
            @test all(0 .< v3)

            v4 = randexp(T{0, Float32})
            @test v4 isa T{0, Float32}
            @test all(0 .< v4)
        end
    end

    @testset "randexp!()" begin
        for T in (MVector, SizedVector)
            v1 = randexp(T{3})
            randexp!(v1)
            @test v1 isa T{3, Float64}
            @test all(0 .< v1)

            v2 = randexp(T{0})
            randexp!(v2)
            @test v2 isa T{0, Float64}
            @test all(0 .< v2)

            v3 = randexp(T{3, Float32})
            randexp!(v3)
            @test v3 isa T{3, Float32}
            @test all(0 .< v3)

            v4 = randexp(T{0, Float32})
            randexp!(v4)
            @test v4 isa T{0, Float32}
            @test all(0 .< v4)
        end
    end

    @testset "arithmetic_closure" for T0 in [subtypes(Unsigned);
                                             subtypes(Signed);
                                             subtypes(AbstractFloat);
                                             Bool;
                                             Complex{Int};
                                             Complex{Float64};
                                             BigInt
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
