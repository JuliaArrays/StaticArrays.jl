@testset "MMatrix" begin
    @testset "Inner Constructors" begin
        @test MMatrix{1,1,Int,1}((1,)).data === (1,)
        @test MMatrix{1,1,Float64,1}((1,)).data === (1.0,)
        @test MMatrix{2,2,Float64,4}((1, 1.0, 1, 1)).data === (1.0, 1.0, 1.0, 1.0)
        @test isa(MMatrix{1,1,Int,1}(undef), MMatrix{1,1,Int,1})
        @test isa(MMatrix{1,1,Int}(undef), MMatrix{1,1,Int,1})

        # Bad input
        @test_throws Exception MMatrix{2,1,Int,2}((1,))
        @test_throws Exception MMatrix{1,1,Int,1}(())

        # Bad parameters
        @test_throws Exception MMatrix{1,1,Int,2}((1,))
        @test_throws Exception MMatrix{1,1,1,1}((1,))
        @test_throws ArgumentError MMatrix{1,2,Int,1}((1,))
        @test_throws ArgumentError MMatrix{2,1,Int,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        # The following tests are much similar in `test/SMatrix.jl`
        @test_throws Exception MMatrix(1,2,3,4) # unknown constructor

        @test MMatrix{1,1,Int}((1,)).data === (1,)
        @test MMatrix{1,1}((1,)).data === (1,)
        @test MMatrix{1}((1,)).data === (1,)

        @test MMatrix{2,2,Int}((1,2,3,4)).data === (1,2,3,4)
        @test MMatrix{2,2}((1,2,3,4)).data === (1,2,3,4)
        @test MMatrix{2}((1,2,3,4)).data === (1,2,3,4)
        @test_throws DimensionMismatch MMatrix{2}((1,2,3,4,5))

        # test for #557-like issues
        @test (@inferred MMatrix(SMatrix{0,0,Float64}()))::MMatrix{0,0,Float64} == MMatrix{0,0,Float64}()

        @test (MMatrix{2,3}(i+10j for i in 1:2, j in 1:3)::MMatrix{2,3}).data ===
            (11,12,21,22,31,32)
        @test (MMatrix{2,3}(float(i+10j) for i in 1:2, j in 1:3)::MMatrix{2,3}).data ===
            (11.0,12.0,21.0,22.0,31.0,32.0)
        @test (MMatrix{0,0,Int}()::MMatrix{0,0}).data === ()
        @test (MMatrix{0,3,Int}()::MMatrix{0,3}).data === ()
        @test (MMatrix{2,0,Int}()::MMatrix{2,0}).data === ()
        @test (MMatrix{2,3,Int}(i+10j for i in 1:2, j in 1:3)::MMatrix{2,3}).data ===
            (11,12,21,22,31,32)
        @test (MMatrix{2,3,Float64}(i+10j for i in 1:2, j in 1:3)::MMatrix{2,3}).data ===
            (11.0,12.0,21.0,22.0,31.0,32.0)
        @test_throws Exception MMatrix{2,3}(i+10j for i in 1:1, j in 1:3)
        @test_throws Exception MMatrix{2,3}(i+10j for i in 1:3, j in 1:3)
        @test_throws Exception MMatrix{2,3,Int}(i+10j for i in 1:1, j in 1:3)
        @test_throws Exception MMatrix{2,3,Int}(i+10j for i in 1:3, j in 1:3)

        @test ((@MMatrix [1.0])::MMatrix{1,1}).data === (1.0,)
        @test ((@MMatrix [1 2])::MMatrix{1,2}).data === (1, 2)
        @test ((@MMatrix [1 ; 2])::MMatrix{2,1}).data === (1, 2)
        @test ((@MMatrix [1 2 ; 3 4])::MMatrix{2,2}).data === (1, 3, 2, 4)

        @test ((@MMatrix Int[1.0])::MMatrix{1,1}).data === (1,)
        @test ((@MMatrix Float64[1 2])::MMatrix{1,2}).data === (1.0, 2.0)
        @test ((@MMatrix Float64[1 ; 2])::MMatrix{2,1}).data === (1.0, 2.0)
        @test ((@MMatrix Float64[1 2 ; 3 4])::MMatrix{2,2}).data === (1.0, 3.0, 2.0, 4.0)

        @test ((@MMatrix [i*j for i = 1:2, j=2:3])::MMatrix{2,2}).data === (2, 4, 3, 6)
        @test ((@MMatrix Float64[i*j for i = 1:2, j=2:3])::MMatrix{2,2}).data === (2.0, 4.0, 3.0, 6.0)

        @testset "expand error" begin
            test_expand_error(:(@MMatrix [1 2; 3]))
            test_expand_error(:(@MMatrix Float32[1 2; 3]))
            test_expand_error(:(@MMatrix [i*j*k for i = 1:2, j=2:3, k=3:4]))
            test_expand_error(:(@MMatrix Float32[i*j*k for i = 1:2, j=2:3, k=3:4]))
            test_expand_error(:(@MMatrix fill(2.3, 4, 5, 6)))
            test_expand_error(:(@MMatrix ones(4, 5, 6, 7)))
            test_expand_error(:(@MMatrix ones))
            test_expand_error(:(@MMatrix sin(1:5)))
            test_expand_error(:(@MMatrix [1; 2; 3; 4]...))
            test_expand_error(:(@MMatrix a))
        end

        @test ((@MMatrix [1 2.;3 4])::MMatrix{2, 2, Float64}).data === (1., 3., 2., 4.) #issue #911
        @test ((@MMatrix fill(3.4, 2,2))::MMatrix{2, 2, Float64}).data === (3.4, 3.4, 3.4, 3.4)
        @test ((@MMatrix zeros(2,2))::MMatrix{2, 2, Float64}).data === (0.0, 0.0, 0.0, 0.0)
        @test ((@MMatrix ones(2,2))::MMatrix{2, 2, Float64}).data === (1.0, 1.0, 1.0, 1.0)
        @test ((@MMatrix zeros(Float32, 2, 2))::MMatrix{2,2,Float32}).data === (0.0f0, 0.0f0, 0.0f0, 0.0f0)
        @test ((@MMatrix ones(Float32, 2, 2))::MMatrix{2,2,Float32}).data === (1.0f0, 1.0f0, 1.0f0, 1.0f0)

        @testset "@MMatrix rand*" begin
            n = 4
            @testset "zero-length" begin
                @test (@MMatrix rand(0, 0)) isa MMatrix{0, 0, Float64}
                @test (@MMatrix rand(0, n)) isa MMatrix{0, n, Float64}
                @test (@MMatrix rand(n, 0)) isa MMatrix{n, 0, Float64}
                @test (@MMatrix rand(Float32, 0, 0)) isa MMatrix{0, 0, Float32}
                @test (@MMatrix rand(Float32, 0, n)) isa MMatrix{0, n, Float32}
                @test (@MMatrix rand(Float32, n, 0)) isa MMatrix{n, 0, Float32}
                @test (@MMatrix rand(_rng(), Float32, 0, 0)) isa MMatrix{0, 0, Float32}
                @test (@MMatrix rand(_rng(), Float32, 0, n)) isa MMatrix{0, n, Float32}
                @test (@MMatrix rand(_rng(), Float32, n, 0)) isa MMatrix{n, 0, Float32}
            end

            @test (@MMatrix rand(n, n)) isa MMatrix{n, n, Float64}
            @test (@MMatrix randn(n, n)) isa MMatrix{n, n, Float64}
            @test (@MMatrix randexp(n, n)) isa MMatrix{n, n, Float64}
            @test (@MMatrix rand(4, 4)) isa MMatrix{4, 4, Float64}
            @test (@MMatrix randn(4, 4)) isa MMatrix{4, 4, Float64}
            @test (@MMatrix randexp(4, 4)) isa MMatrix{4, 4, Float64}
            @test (@MMatrix rand(_rng(), n, n)) isa MMatrix{n, n, Float64}
            @test (@MMatrix rand(_rng(), n, n)) == rand(_rng(), n, n)
            @test (@MMatrix randn(_rng(), n, n)) isa MMatrix{n, n, Float64}
            @test (@MMatrix randn(_rng(), n, n)) == randn(_rng(), n, n)
            @test (@MMatrix randexp(_rng(), n, n)) isa MMatrix{n, n, Float64}
            @test (@MMatrix randexp(_rng(), n, n)) == randexp(_rng(), n, n)
            @test (@MMatrix rand(_rng(), 4, 4)) isa MMatrix{4, 4, Float64}
            @test (@MMatrix rand(_rng(), 4, 4)) == rand(_rng(), 4, 4)
            @test (@MMatrix randn(_rng(), 4, 4)) isa MMatrix{4, 4, Float64}
            @test (@MMatrix randn(_rng(), 4, 4)) == randn(_rng(), 4, 4)
            @test (@MMatrix randexp(_rng(), 4, 4)) isa MMatrix{4, 4, Float64}
            @test (@MMatrix randexp(_rng(), 4, 4)) == randexp(_rng(), 4, 4)

            for T in (Float32, Float64)
                @test (@MMatrix rand(T, n, n)) isa MMatrix{n, n, T}
                @test (@MMatrix randn(T, n, n)) isa MMatrix{n, n, T}
                @test (@MMatrix randexp(T, n, n)) isa MMatrix{n, n, T}
                @test (@MMatrix rand(T, 4, 4)) isa MMatrix{4, 4, T}
                @test (@MMatrix randn(T, 4, 4)) isa MMatrix{4, 4, T}
                @test (@MMatrix randexp(T, 4, 4)) isa MMatrix{4, 4, T}
                @test (@MMatrix rand(_rng(), T, n, n)) isa MMatrix{n, n, T}
                VERSION≥v"1.7" && @test (@MMatrix rand(_rng(), T, n, n)) == rand(_rng(), T, n, n) broken=(T===Float32)
                @test (@MMatrix randn(_rng(), T, n, n)) isa MMatrix{n, n, T}
                @test (@MMatrix randn(_rng(), T, n, n)) == randn(_rng(), T, n, n)
                @test (@MMatrix randexp(_rng(), T, n, n)) isa MMatrix{n, n, T}
                @test (@MMatrix randexp(_rng(), T, n, n)) == randexp(_rng(), T, n, n)
                @test (@MMatrix rand(_rng(), T, 4, 4)) isa MMatrix{4, 4, T}
                VERSION≥v"1.7" && @test (@MMatrix rand(_rng(), T, 4, 4)) == rand(_rng(), T, 4, 4) broken=(T===Float32)
                @test (@MMatrix randn(_rng(), T, 4, 4)) isa MMatrix{4, 4, T}
                @test (@MMatrix randn(_rng(), T, 4, 4)) == randn(_rng(), T, 4, 4)
                @test (@MMatrix randexp(_rng(), T, 4, 4)) isa MMatrix{4, 4, T}
                @test (@MMatrix randexp(_rng(), T, 4, 4)) == randexp(_rng(), T, 4, 4)
            end
        end

        @inferred MMatrix(rand(MMatrix{3, 3})) # issue 356
        @test MMatrix(SMatrix{1,1,Int,1}((1,))).data == (1,)
        @test_throws DimensionMismatch MMatrix{3}((1,2,3,4))

        if VERSION >= v"1.7.0"
            @test ((@MMatrix Float64[1;2;3;;;])::MMatrix{3,1}).data === (1.0, 2.0, 3.0)
            @test ((@MMatrix [1;2;3;;;])::MMatrix{3,1}).data === (1, 2, 3)
            test_expand_error(:(@MMatrix [1;2;;;1;2]))
        end
    end

    @testset "Methods" begin
        m = @MMatrix [11 13; 12 14]

        @test isimmutable(m) == false

        @test m[1] === 11
        @test m[2] === 12
        @test m[3] === 13
        @test m[4] === 14

        @testinf Tuple(m) === (11, 12, 13, 14)

        @test size(m) === (2, 2)
        @test size(typeof(m)) === (2, 2)
        @test size(MMatrix{2,2}) === (2, 2)

        @test size(m, 1) === 2
        @test size(m, 2) === 2
        @test size(typeof(m), 1) === 2
        @test size(typeof(m), 2) === 2

        @test length(m) === 4

        @test reverse(m) == reverse(reverse(collect(m), dims = 2), dims = 1)
    end

    @testset "setindex!" begin
        m = @MMatrix [0 0; 0 0]
        m[1] = 11
        m[2] = 12
        m[3] = 13
        m[4] = 14
        @test m.data === (11, 12, 13, 14)
        @test setindex!(m, 13, 3) === m
        @test setindex!(m, 12, 2, 1) === m

        m = @MMatrix [0 0; 0 0]
        m[1] = Int8(11)
        m[2] = Int8(12)
        m[3] = Int8(13)
        m[4] = Int8(14)
        @test m.data === (11, 12, 13, 14)

        # setindex with non-elbits type
        m = MMatrix{2,2,String}(undef)
        @test_throws ErrorException setindex!(m, "a", 1, 1)
    end
end
