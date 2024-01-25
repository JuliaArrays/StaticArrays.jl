@testset "SMatrix" begin
    @testset "Inner Constructors" begin
        @test SMatrix{1,1,Int,1}((1,)).data === (1,)
        @test SMatrix{1,1,Float64,1}((1,)).data === (1.0,)
        @test SMatrix{2,2,Float64,4}((1, 1.0, 1, 1)).data === (1.0, 1.0, 1.0, 1.0)

        # Bad input
        @test_throws Exception SMatrix{1,1,Int,1}()
        @test_throws Exception SMatrix{2,1,Int,2}((1,))
        @test_throws Exception SMatrix{1,1,Int,1}(())

        # Bad parameters
        @test_throws Exception SMatrix{1,1,Int,2}((1,))
        @test_throws ArgumentError SMatrix{1,1,1,1}((1,))
        @test_throws ArgumentError SMatrix{1,2,Int,1}((1,))
        @test_throws ArgumentError SMatrix{2,1,Int,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        # The following tests are much similar in `test/MMatrix.jl`
        @test_throws Exception SMatrix(1,2,3,4) # unknown constructor

        @test SMatrix{1,1,Int}((1,)).data === (1,)
        @test SMatrix{1,1}((1,)).data === (1,)
        @test SMatrix{1}((1,)).data === (1,)

        @test SMatrix{2,2,Int}((1,2,3,4)).data === (1,2,3,4)
        @test SMatrix{2,2}((1,2,3,4)).data === (1,2,3,4)
        @test SMatrix{2}((1,2,3,4)).data === (1,2,3,4)
        @test_throws DimensionMismatch SMatrix{2}((1,2,3,4,5))

        # test for #557-like issues
        @test (@inferred SMatrix(MMatrix{0,0,Float64}()))::SMatrix{0,0,Float64} == SMatrix{0,0,Float64}()

        @test (SMatrix{2,3}(i+10j for i in 1:2, j in 1:3)::SMatrix{2,3}).data ===
            (11,12,21,22,31,32)
        @test (SMatrix{2,3}(float(i+10j) for i in 1:2, j in 1:3)::SMatrix{2,3}).data ===
            (11.0,12.0,21.0,22.0,31.0,32.0)
        @test (SMatrix{0,0,Int}()::SMatrix{0,0}).data === ()
        @test (SMatrix{0,3,Int}()::SMatrix{0,3}).data === ()
        @test (SMatrix{2,0,Int}()::SMatrix{2,0}).data === ()
        @test (SMatrix{2,3,Int}(i+10j for i in 1:2, j in 1:3)::SMatrix{2,3}).data ===
            (11,12,21,22,31,32)
        @test (SMatrix{2,3,Float64}(i+10j for i in 1:2, j in 1:3)::SMatrix{2,3}).data ===
            (11.0,12.0,21.0,22.0,31.0,32.0)
        @test_throws Exception SMatrix{2,3}(i+10j for i in 1:1, j in 1:3)
        @test_throws Exception SMatrix{2,3}(i+10j for i in 1:3, j in 1:3)
        @test_throws Exception SMatrix{2,3,Int}(i+10j for i in 1:1, j in 1:3)
        @test_throws Exception SMatrix{2,3,Int}(i+10j for i in 1:3, j in 1:3)

        @test ((@SMatrix [1.0])::SMatrix{1,1}).data === (1.0,)
        @test ((@SMatrix [1 2])::SMatrix{1,2}).data === (1, 2)
        @test ((@SMatrix [1 ; 2])::SMatrix{2,1}).data === (1, 2)
        @test ((@SMatrix [1 2 ; 3 4])::SMatrix{2,2}).data === (1, 3, 2, 4)

        @test ((@SMatrix Int[1.0])::SMatrix{1,1}).data === (1,)
        @test ((@SMatrix Float64[1 2])::SMatrix{1,2}).data === (1.0, 2.0)
        @test ((@SMatrix Float64[1 ; 2])::SMatrix{2,1}).data === (1.0, 2.0)
        @test ((@SMatrix Float64[1 2 ; 3 4])::SMatrix{2,2}).data === (1.0, 3.0, 2.0, 4.0)

        @test ((@SMatrix [i*j for i = 1:2, j=2:3])::SMatrix{2,2}).data === (2, 4, 3, 6)
        @test ((@SMatrix Float64[i*j for i = 1:2, j=2:3])::SMatrix{2,2}).data === (2.0, 4.0, 3.0, 6.0)

        @testset "expand error" begin
            test_expand_error(:(@SMatrix [1 2; 3]))
            test_expand_error(:(@SMatrix Float64[1 2; 3]))
            test_expand_error(:(@SMatrix [i*j*k for i = 1:2, j=2:3, k=3:4]))
            test_expand_error(:(@SMatrix Float64[i*j*k for i = 1:2, j=2:3, k=3:4]))
            test_expand_error(:(@SMatrix fill(1.5, 2, 3, 4)))
            test_expand_error(:(@SMatrix ones(2, 3, 4, 5)))
            test_expand_error(:(@SMatrix ones))
            test_expand_error(:(@SMatrix sin(1:5)))
            test_expand_error(:(@SMatrix [1; 2; 3; 4]...))
            test_expand_error(:(@SMatrix a))
        end

        @test ((@SMatrix [1 2.;3 4])::SMatrix{2, 2, Float64}).data === (1., 3., 2., 4.) #issue #911
        @test ((@SMatrix fill(1.3, 2,2))::SMatrix{2, 2, Float64}).data === (1.3, 1.3, 1.3, 1.3)
        @test ((@SMatrix zeros(2,2))::SMatrix{2, 2, Float64}).data === (0.0, 0.0, 0.0, 0.0)
        @test ((@SMatrix ones(2,2))::SMatrix{2, 2, Float64}).data === (1.0, 1.0, 1.0, 1.0)
        @test ((@SMatrix zeros(Float32, 2, 2))::SMatrix{2,2,Float32}).data === (0.0f0, 0.0f0, 0.0f0, 0.0f0)
        @test ((@SMatrix ones(Float32, 2, 2))::SMatrix{2,2,Float32}).data === (1.0f0, 1.0f0, 1.0f0, 1.0f0)

        @testset "@SMatrix rand*" begin
            n = 4
            @testset "zero-length" begin
                @test (@SMatrix rand(0, 0)) isa SMatrix{0, 0, Float64}
                @test (@SMatrix rand(0, n)) isa SMatrix{0, n, Float64}
                @test (@SMatrix rand(n, 0)) isa SMatrix{n, 0, Float64}
                @test (@SMatrix rand(Float32, 0, 0)) isa SMatrix{0, 0, Float32}
                @test (@SMatrix rand(Float32, 0, n)) isa SMatrix{0, n, Float32}
                @test (@SMatrix rand(Float32, n, 0)) isa SMatrix{n, 0, Float32}
                @test (@SMatrix rand(_rng(), Float32, 0, 0)) isa SMatrix{0, 0, Float32}
                @test (@SMatrix rand(_rng(), Float32, 0, n)) isa SMatrix{0, n, Float32}
                @test (@SMatrix rand(_rng(), Float32, n, 0)) isa SMatrix{n, 0, Float32}
            end

            @test (@SMatrix rand(n, n)) isa SMatrix{n, n, Float64}
            @test (@SMatrix randn(n, n)) isa SMatrix{n, n, Float64}
            @test (@SMatrix randexp(n, n)) isa SMatrix{n, n, Float64}
            @test (@SMatrix rand(4, 4)) isa SMatrix{4, 4, Float64}
            @test (@SMatrix randn(4, 4)) isa SMatrix{4, 4, Float64}
            @test (@SMatrix randexp(4, 4)) isa SMatrix{4, 4, Float64}
            @test (@SMatrix rand(_rng(), n, n)) isa SMatrix{n, n, Float64}
            @test (@SMatrix rand(_rng(), n, n)) == rand(_rng(), n, n)
            @test (@SMatrix randn(_rng(), n, n)) isa SMatrix{n, n, Float64}
            @test (@SMatrix randn(_rng(), n, n)) == randn(_rng(), n, n)
            @test (@SMatrix randexp(_rng(), n, n)) isa SMatrix{n, n, Float64}
            @test (@SMatrix randexp(_rng(), n, n)) == randexp(_rng(), n, n)
            @test (@SMatrix rand(_rng(), 4, 4)) isa SMatrix{4, 4, Float64}
            @test (@SMatrix rand(_rng(), 4, 4)) == rand(_rng(), 4, 4)
            @test (@SMatrix randn(_rng(), 4, 4)) isa SMatrix{4, 4, Float64}
            @test (@SMatrix randn(_rng(), 4, 4)) == randn(_rng(), 4, 4)
            @test (@SMatrix randexp(_rng(), 4, 4)) isa SMatrix{4, 4, Float64}
            @test (@SMatrix randexp(_rng(), 4, 4)) == randexp(_rng(), 4, 4)

            for T in (Float32, Float64)
                @test (@SMatrix rand(T, n, n)) isa SMatrix{n, n, T}
                @test (@SMatrix randn(T, n, n)) isa SMatrix{n, n, T}
                @test (@SMatrix randexp(T, n, n)) isa SMatrix{n, n, T}
                @test (@SMatrix rand(T, 4, 4)) isa SMatrix{4, 4, T}
                @test (@SMatrix randn(T, 4, 4)) isa SMatrix{4, 4, T}
                @test (@SMatrix randexp(T, 4, 4)) isa SMatrix{4, 4, T}
                @test (@SMatrix rand(_rng(), T, n, n)) isa SMatrix{n, n, T}
                VERSION≥v"1.7" && @test (@SMatrix rand(_rng(), T, n, n)) == rand(_rng(), T, n, n) broken=(T===Float32)
                @test (@SMatrix randn(_rng(), T, n, n)) isa SMatrix{n, n, T}
                @test (@SMatrix randn(_rng(), T, n, n)) == randn(_rng(), T, n, n)
                @test (@SMatrix randexp(_rng(), T, n, n)) isa SMatrix{n, n, T}
                @test (@SMatrix randexp(_rng(), T, n, n)) == randexp(_rng(), T, n, n)
                @test (@SMatrix rand(_rng(), T, 4, 4)) isa SMatrix{4, 4, T}
                VERSION≥v"1.7" && @test (@SMatrix rand(_rng(), T, 4, 4)) == rand(_rng(), T, 4, 4) broken=(T===Float32)
                @test (@SMatrix randn(_rng(), T, 4, 4)) isa SMatrix{4, 4, T}
                @test (@SMatrix randn(_rng(), T, 4, 4)) == randn(_rng(), T, 4, 4)
                @test (@SMatrix randexp(_rng(), T, 4, 4)) isa SMatrix{4, 4, T}
                @test (@SMatrix randexp(_rng(), T, 4, 4)) == randexp(_rng(), T, 4, 4)
            end
        end

        @inferred SMatrix(rand(SMatrix{3, 3})) # issue 356
        @test SMatrix(MMatrix{1,1,Int,1}((1,))).data == (1,)
        @test_throws DimensionMismatch SMatrix{3}((1,2,3,4))

        if VERSION >= v"1.7.0"
            @test ((@SMatrix Float64[1;2;3;;;])::SMatrix{3,1}).data === (1.0, 2.0, 3.0)
            @test ((@SMatrix [1;2;3;;;])::SMatrix{3,1}).data === (1, 2, 3)
            test_expand_error(:(@SMatrix [1;2;;;1;2]))
        end
    end

    @testset "Methods" begin
        m = @SMatrix [11 13; 12 14]

        @test isimmutable(m) == true

        @test m[1] === 11
        @test m[2] === 12
        @test m[3] === 13
        @test m[4] === 14

        @testinf Tuple(m) === (11, 12, 13, 14)

        @test size(m) === (2, 2)
        @test size(typeof(m)) === (2, 2)
        @test size(SMatrix{2,2}) === (2, 2)

        @test size(m, 1) === 2
        @test size(m, 2) === 2
        @test size(typeof(m), 1) === 2
        @test size(typeof(m), 2) === 2

        @test length(m) === 4

        @test_throws Exception m[1] = 1

        @test reverse(m) == reverse(reverse(collect(m), dims = 2), dims = 1)
    end
end
