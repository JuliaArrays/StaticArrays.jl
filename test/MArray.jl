@testset "MArray" begin
    @testset "Inner Constructors" begin
        @test MArray{Tuple{1},Int,1,1}((1,)).data === (1,)
        @test MArray{Tuple{1},Float64,1,1}((1,)).data === (1.0,)
        @test MArray{Tuple{2,2},Float64,2,4}((1, 1.0, 1, 1)).data === (1.0, 1.0, 1.0, 1.0)
        @test isa(MArray{Tuple{1},Int,1,1}(undef), MArray{Tuple{1},Int,1,1})
        @test isa(MArray{Tuple{1},Int,1}(undef), MArray{Tuple{1},Int,1,1})
        @test isa(MArray{Tuple{1},Int}(undef), MArray{Tuple{1},Int,1,1})

        # Bad input
        @test_throws Exception MArray{Tuple{2},Int,1,2}((1,))
        @test_throws Exception MArray{Tuple{1},Int,1,1}(())

        # Bad parameters
        @test_throws Exception MArray{Tuple{1},Int,1,2}((1,))
        @test_throws Exception MArray{Tuple{1},Int,2,1}((1,))
        @test_throws Exception MArray{Tuple{1},1,1,1}((1,))
        @test_throws Exception MArray{Tuple{2},Int,1,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        @test_throws Exception MArray(1,2,3,4) # unknown constructor

        @test MArray{Tuple{1},Int,1}((1,)).data === (1,)
        @test MArray{Tuple{1},Int}((1,)).data === (1,)
        @test MArray{Tuple{1}}((1,)).data === (1,)

        @test MArray{Tuple{2,2},Int,2}((1,2,3,4)).data === (1,2,3,4)
        @test MArray{Tuple{2,2},Int}((1,2,3,4)).data === (1,2,3,4)
        @test MArray{Tuple{2,2}}((1,2,3,4)).data === (1,2,3,4)

        @test MArray(SVector(1,2)) isa MArray{Tuple{2}}
        # Constructors should create a copy (#335)
        v = MArray{Tuple{2}}(1,2)
        @test MArray(v) !== v && MArray(v) == v

        # test for #557-like issues
        @test (@inferred MArray(SVector{0,Float64}()))::MVector{0,Float64} == MVector{0,Float64}()

        @test MArray{Tuple{}}(i for i in 1:1).data === (1,)
        @test MArray{Tuple{3}}(i for i in 1:3).data === (1,2,3)
        @test MArray{Tuple{3}}(float(i) for i in 1:3).data === (1.0,2.0,3.0)
        @test MArray{Tuple{2,3}}(i+10j for i in 1:2, j in 1:3).data === (11,12,21,22,31,32)
        @test MArray{Tuple{1,2,3}}(i+10j+100k for i in 1:1, j in 1:2, k in 1:3).data === (111,121,211,221,311,321)
        @test_throws Exception MArray{Tuple{}}(i for i in 1:0)
        @test_throws Exception MArray{Tuple{}}(i for i in 1:2)
        @test_throws Exception MArray{Tuple{3}}(i for i in 1:2)
        @test_throws Exception MArray{Tuple{3}}(i for i in 1:4)
        @test_throws Exception MArray{Tuple{2,3}}(10i+j for i in 1:1, j in 1:3)
        @test_throws Exception MArray{Tuple{2,3}}(10i+j for i in 1:3, j in 1:3)

        @test StaticArrays.sacollect(MVector{6}, Iterators.product(1:2, 1:3)) ==
            MVector{6}(collect(Iterators.product(1:2, 1:3)))
        @test StaticArrays.sacollect(MVector{2}, Iterators.zip(1:2, 2:3)) ==
            MVector{2}(collect(Iterators.zip(1:2, 2:3)))
        @test StaticArrays.sacollect(MVector{3}, Iterators.take(1:10, 3)) ==
            MVector{3}(collect(Iterators.take(1:10, 3)))
        @test StaticArrays.sacollect(MMatrix{2,3}, Iterators.product(1:2, 1:3)) ==
            MMatrix{2,3}(collect(Iterators.product(1:2, 1:3)))
        @test StaticArrays.sacollect(MArray{Tuple{2,3,4}}, 1:24) ==
            MArray{Tuple{2,3,4}}(collect(1:24))

        @test ((@MArray [1])::MArray{Tuple{1}}).data === (1,)
        @test ((@MArray [1,2])::MArray{Tuple{2}}).data === (1,2)
        @test ((@MArray Float64[1,2,3])::MArray{Tuple{3}}).data === (1.0, 2.0, 3.0)
        @test ((@MArray [1 2])::MArray{Tuple{1,2}}).data === (1, 2)
        @test ((@MArray Float64[1 2])::MArray{Tuple{1,2}}).data === (1.0, 2.0)
        @test ((@MArray [1 ; 2])::MArray{Tuple{2}}).data === (1, 2)
        @test ((@MArray Float64[1 ; 2])::MArray{Tuple{2}}).data === (1.0, 2.0)
        @test ((@MArray [1 2 ; 3 4])::MArray{Tuple{2,2}}).data === (1, 3, 2, 4)
        @test ((@MArray Float64[1 2 ; 3 4])::MArray{Tuple{2,2}}).data === (1.0, 3.0, 2.0, 4.0)

        @test ((@MArray [i for i = 1:2])::MArray{Tuple{2}}).data === (1, 2)
        @test ((@MArray [i*j for i = 1:2, j = 2:3])::MArray{Tuple{2,2}}).data === (2, 4, 3, 6)
        @test ((@MArray [i*j*k for i = 1:2, j = 2:3, k = 3:4])::MArray{Tuple{2,2,2}}).data === (6, 12, 9, 18, 8, 16, 12, 24)
        @test ((@MArray [i*j*k*l for i = 1:2, j = 2:3, k = 3:4, l = 1:2])::MArray{Tuple{2,2,2,2}}).data === (6, 12, 9, 18, 8, 16, 12, 24, 12, 24, 18, 36, 16, 32, 24, 48)
        @test ((@MArray [i*j*k*l*m for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2])::MArray{Tuple{2,2,2,2,2}}).data === (6, 12, 9, 18, 8, 16, 12, 24, 12, 24, 18, 36, 16, 32, 24, 48, 2*6, 2*12, 2*9, 2*18, 2*8, 2*16, 2*12, 2*24, 2*12, 2*24, 2*18, 2*36, 2*16, 2*32, 2*24, 2*48)
        @test ((@MArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2])::MArray{Tuple{2,2,2,2,2,2}}).data === ntuple(i->1, 64)
        @test ((@MArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2])::MArray{Tuple{2,2,2,2,2,2,2}}).data === ntuple(i->1, 128)
        @test ((@MArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2])::MArray{Tuple{2,2,2,2,2,2,2,2}}).data === ntuple(i->1, 256)
        @test ((@MArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2, q = 1:2])::MArray{Tuple{2,2,2,2,2,2,2,2,2}}).data === ntuple(i->1, 512)
        @test ((@MArray Float64[i for i = 1:2])::MArray{Tuple{2}}).data === (1.0, 2.0)
        @test ((@MArray Float64[i*j for i = 1:2, j = 2:3])::MArray{Tuple{2,2}}).data === (2.0, 4.0, 3.0, 6.0)
        @test ((@MArray Float64[i*j*k for i = 1:2, j = 2:3, k =3:4])::MArray{Tuple{2,2,2}}).data === (6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0)
        @test ((@MArray Float64[i*j*k*l for i = 1:2, j = 2:3, k = 3:4, l = 1:2])::MArray{Tuple{2,2,2,2}}).data === (6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0, 12.0, 24.0, 18.0, 36.0, 16.0, 32.0, 24.0, 48.0)
        @test ((@MArray Float64[i*j*k*l*m for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2])::MArray{Tuple{2,2,2,2,2}}).data === (6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0, 12.0, 24.0, 18.0, 36.0, 16.0, 32.0, 24.0, 48.0, 2*6.0, 2*12.0, 2*9.0, 2*18.0, 2*8.0, 2*16.0, 2*12.0, 2*24.0, 2*12.0, 2*24.0, 2*18.0, 2*36.0, 2*16.0, 2*32.0, 2*24.0, 2*48.0)
        @test ((@MArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2])::MArray{Tuple{2,2,2,2,2,2}}).data === ntuple(i->1.0, 64)
        @test ((@MArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2])::MArray{Tuple{2,2,2,2,2,2,2}}).data === ntuple(i->1.0, 128)
        @test ((@MArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2])::MArray{Tuple{2,2,2,2,2,2,2,2}}).data === ntuple(i->1.0, 256)
        @test ((@MArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2, q = 1:2])::MArray{Tuple{2,2,2,2,2,2,2,2,2}}).data === ntuple(i->1.0, 512)

        @testset "expand error" begin
            test_expand_error(:(@MArray [1 2; 3]))
            test_expand_error(:(@MArray Float64[1 2; 3]))
            test_expand_error(:(@MArray fill))
            test_expand_error(:(@MArray ones))
            test_expand_error(:(@MArray sin(1:5)))
            test_expand_error(:(@MArray fill()))
            test_expand_error(:(@MArray [1; 2; 3; 4]...))

            # (typed-)comprehension LoadError for `ex.args[1].head != :generator`
            test_expand_error(:(@MArray [i+j for i in 1:2 for j in 1:2]))
            test_expand_error(:(@MArray Int[i+j for i in 1:2 for j in 1:2]))
        end

        @testset "@MArray rand*" begin
            @testset "Same test as @MVector rand*" begin
                n = 4
                @test (@MArray rand(n)) isa MVector{n, Float64}
                @test (@MArray randn(n)) isa MVector{n, Float64}
                @test (@MArray randexp(n)) isa MVector{n, Float64}
                @test (@MArray rand(4)) isa MVector{4, Float64}
                @test (@MArray randn(4)) isa MVector{4, Float64}
                @test (@MArray randexp(4)) isa MVector{4, Float64}
                @test (@MArray rand(_rng(), n)) isa MVector{n, Float64}
                @test (@MArray rand(_rng(), n)) == rand(_rng(), n)
                @test (@MArray randn(_rng(), n)) isa MVector{n, Float64}
                @test (@MArray randn(_rng(), n)) == randn(_rng(), n)
                @test (@MArray randexp(_rng(), n)) isa MVector{n, Float64}
                @test (@MArray randexp(_rng(), n)) == randexp(_rng(), n)
                @test (@MArray rand(_rng(), 4)) isa MVector{4, Float64}
                @test (@MArray rand(_rng(), 4)) == rand(_rng(), 4)
                @test (@MArray randn(_rng(), 4)) isa MVector{4, Float64}
                @test (@MArray randn(_rng(), 4)) == randn(_rng(), 4)
                @test (@MArray randexp(_rng(), 4)) isa MVector{4, Float64}
                @test (@MArray randexp(_rng(), 4)) == randexp(_rng(), 4)

                for T in (Float32, Float64)
                    @test (@MArray rand(T, n)) isa MVector{n, T}
                    @test (@MArray randn(T, n)) isa MVector{n, T}
                    @test (@MArray randexp(T, n)) isa MVector{n, T}
                    @test (@MArray rand(T, 4)) isa MVector{4, T}
                    @test (@MArray randn(T, 4)) isa MVector{4, T}
                    @test (@MArray randexp(T, 4)) isa MVector{4, T}
                    @test (@MArray rand(_rng(), T, n)) isa MVector{n, T}
                    VERSION≥v"1.7" && @test (@MArray rand(_rng(), T, n)) == rand(_rng(), T, n) broken=(T===Float32)
                    @test (@MArray randn(_rng(), T, n)) isa MVector{n, T}
                    @test (@MArray randn(_rng(), T, n)) == randn(_rng(), T, n)
                    @test (@MArray randexp(_rng(), T, n)) isa MVector{n, T}
                    @test (@MArray randexp(_rng(), T, n)) == randexp(_rng(), T, n)
                    @test (@MArray rand(_rng(), T, 4)) isa MVector{4, T}
                    VERSION≥v"1.7" && @test (@MArray rand(_rng(), T, 4)) == rand(_rng(), T, 4) broken=(T===Float32)
                    @test (@MArray randn(_rng(), T, 4)) isa MVector{4, T}
                    @test (@MArray randn(_rng(), T, 4)) == randn(_rng(), T, 4)
                    @test (@MArray randexp(_rng(), T, 4)) isa MVector{4, T}
                    @test (@MArray randexp(_rng(), T, 4)) == randexp(_rng(), T, 4)
                end
            end

            @testset "Same tests as @MMatrix rand*" begin
                n = 4
                @testset "zero-length" begin
                    @test (@MArray rand(0, 0)) isa MMatrix{0, 0, Float64}
                    @test (@MArray rand(0, n)) isa MMatrix{0, n, Float64}
                    @test (@MArray rand(n, 0)) isa MMatrix{n, 0, Float64}
                    @test (@MArray rand(Float32, 0, 0)) isa MMatrix{0, 0, Float32}
                    @test (@MArray rand(Float32, 0, n)) isa MMatrix{0, n, Float32}
                    @test (@MArray rand(Float32, n, 0)) isa MMatrix{n, 0, Float32}
                    @test (@MArray rand(_rng(), Float32, 0, 0)) isa MMatrix{0, 0, Float32}
                    @test (@MArray rand(_rng(), Float32, 0, n)) isa MMatrix{0, n, Float32}
                    @test (@MArray rand(_rng(), Float32, n, 0)) isa MMatrix{n, 0, Float32}
                end

                @test (@MArray rand(n, n)) isa MMatrix{n, n, Float64}
                @test (@MArray randn(n, n)) isa MMatrix{n, n, Float64}
                @test (@MArray randexp(n, n)) isa MMatrix{n, n, Float64}
                @test (@MArray rand(4, 4)) isa MMatrix{4, 4, Float64}
                @test (@MArray randn(4, 4)) isa MMatrix{4, 4, Float64}
                @test (@MArray randexp(4, 4)) isa MMatrix{4, 4, Float64}
                @test (@MArray rand(_rng(), n, n)) isa MMatrix{n, n, Float64}
                @test (@MArray rand(_rng(), n, n)) == rand(_rng(), n, n)
                @test (@MArray randn(_rng(), n, n)) isa MMatrix{n, n, Float64}
                @test (@MArray randn(_rng(), n, n)) == randn(_rng(), n, n)
                @test (@MArray randexp(_rng(), n, n)) isa MMatrix{n, n, Float64}
                @test (@MArray randexp(_rng(), n, n)) == randexp(_rng(), n, n)
                @test (@MArray rand(_rng(), 4, 4)) isa MMatrix{4, 4, Float64}
                @test (@MArray rand(_rng(), 4, 4)) == rand(_rng(), 4, 4)
                @test (@MArray randn(_rng(), 4, 4)) isa MMatrix{4, 4, Float64}
                @test (@MArray randn(_rng(), 4, 4)) == randn(_rng(), 4, 4)
                @test (@MArray randexp(_rng(), 4, 4)) isa MMatrix{4, 4, Float64}
                @test (@MArray randexp(_rng(), 4, 4)) == randexp(_rng(), 4, 4)

                for T in (Float32, Float64)
                    @test (@MArray rand(T, n, n)) isa MMatrix{n, n, T}
                    @test (@MArray randn(T, n, n)) isa MMatrix{n, n, T}
                    @test (@MArray randexp(T, n, n)) isa MMatrix{n, n, T}
                    @test (@MArray rand(T, 4, 4)) isa MMatrix{4, 4, T}
                    @test (@MArray randn(T, 4, 4)) isa MMatrix{4, 4, T}
                    @test (@MArray randexp(T, 4, 4)) isa MMatrix{4, 4, T}
                    @test (@MArray rand(_rng(), T, n, n)) isa MMatrix{n, n, T}
                    VERSION≥v"1.7" && @test (@MArray rand(_rng(), T, n, n)) == rand(_rng(), T, n, n) broken=(T===Float32)
                    @test (@MArray randn(_rng(), T, n, n)) isa MMatrix{n, n, T}
                    @test (@MArray randn(_rng(), T, n, n)) == randn(_rng(), T, n, n)
                    @test (@MArray randexp(_rng(), T, n, n)) isa MMatrix{n, n, T}
                    @test (@MArray randexp(_rng(), T, n, n)) == randexp(_rng(), T, n, n)
                    @test (@MArray rand(_rng(), T, 4, 4)) isa MMatrix{4, 4, T}
                    VERSION≥v"1.7" && @test (@MArray rand(_rng(), T, 4, 4)) == rand(_rng(), T, 4, 4) broken=(T===Float32)
                    @test (@MArray randn(_rng(), T, 4, 4)) isa MMatrix{4, 4, T}
                    @test (@MArray randn(_rng(), T, 4, 4)) == randn(_rng(), T, 4, 4)
                    @test (@MArray randexp(_rng(), T, 4, 4)) isa MMatrix{4, 4, T}
                    @test (@MArray randexp(_rng(), T, 4, 4)) == randexp(_rng(), T, 4, 4)
                end
            end

            @test (@MArray rand(2,2,1))    isa MArray{Tuple{2,2,1}, Float64}
            @test (@MArray rand(2,2,0))    isa MArray{Tuple{2,2,0}, Float64}
            @test (@MArray randn(2,2,1))   isa MArray{Tuple{2,2,1}, Float64}
            @test (@MArray randn(2,2,0))   isa MArray{Tuple{2,2,0}, Float64}
            @test (@MArray randexp(2,2,1)) isa MArray{Tuple{2,2,1}, Float64}
            @test (@MArray randexp(2,2,0)) isa MArray{Tuple{2,2,0}, Float64}
            @test (@MArray rand(Float32,2,2,1))    isa MArray{Tuple{2,2,1}, Float32}
            @test (@MArray rand(Float32,2,2,0))    isa MArray{Tuple{2,2,0}, Float32}
            @test (@MArray randn(Float32,2,2,1))   isa MArray{Tuple{2,2,1}, Float32}
            @test (@MArray randn(Float32,2,2,0))   isa MArray{Tuple{2,2,0}, Float32}
            @test (@MArray randexp(Float32,2,2,1)) isa MArray{Tuple{2,2,1}, Float32}
            @test (@MArray randexp(Float32,2,2,0)) isa MArray{Tuple{2,2,0}, Float32}
        end

        @testset "fill, zeros, ones" begin
            @test ((@MArray fill(1))::MArray{Tuple{},Int}).data === (1,)
            @test ((@MArray zeros())::MArray{Tuple{},Float64}).data === (0.,)
            @test ((@MArray ones())::MArray{Tuple{},Float64}).data === (1.,)
            @test ((@MArray fill(3.,2,2,1))::MArray{Tuple{2,2,1}, Float64}).data === (3.0, 3.0, 3.0, 3.0)
            @test ((@MArray zeros(2,2,1))::MArray{Tuple{2,2,1}, Float64}).data === (0.0, 0.0, 0.0, 0.0)
            @test ((@MArray ones(2,2,1))::MArray{Tuple{2,2,1}, Float64}).data === (1.0, 1.0, 1.0, 1.0)
            @test ((@MArray zeros(3-1,2,1))::MArray{Tuple{2,2,1}, Float64}).data === (0.0, 0.0, 0.0, 0.0)
            @test ((@MArray ones(3-1,2,1))::MArray{Tuple{2,2,1}, Float64}).data === (1.0, 1.0, 1.0, 1.0)
            @test ((@MArray zeros(Float32,2,2,1))::MArray{Tuple{2,2,1}, Float32}).data === (0.f0, 0.f0, 0.f0, 0.f0)
            @test ((@MArray ones(Float32,2,2,1))::MArray{Tuple{2,2,1}, Float32}).data === (1.f0, 1.f0, 1.f0, 1.f0)
            @test ((@MArray zeros(Float32,3-1,2,1))::MArray{Tuple{2,2,1}, Float32}).data === (0.f0, 0.f0, 0.f0, 0.f0)
            @test ((@MArray ones(Float32,3-1,2,1))::MArray{Tuple{2,2,1}, Float32}).data === (1.f0, 1.f0, 1.f0, 1.f0)
        end

        m = [1 2; 3 4]
        @test MArray{Tuple{2,2}}(m) == @MArray [1 2; 3 4]

        # Non-square comprehensions built from SVectors - see #76
        @test @MArray([1 for x = SVector(1,2), y = SVector(1,2,3)]) == ones(2,3)

        # Nested cat
        @test ((@MArray [[1;2] [3;4]])::MMatrix{2,2}).data === (1,2,3,4)
        @test ((@MArray Float64[[1;2] [3;4]])::MMatrix{2,2}).data === (1.,2.,3.,4.)
        @test ((@MArray [[1 3];[2 4]])::MMatrix{2,2}).data === (1,2,3,4)
        @test ((@MArray Float64[[1 3];[2 4]])::MMatrix{2,2}).data === (1.,2.,3.,4.)
        test_expand_error(:(@MArray [[1;2] [3]]))
        test_expand_error(:(@MArray [[1 2]; [3]]))

        @test (@MArray [[[1,2],1]; 2; 3]) == [[[1,2],1]; 2; 3]

        if VERSION >= v"1.7.0"
            function test_ex(ex)
                a = eval(:(@MArray $ex))
                b = eval(ex)
                @test a isa MArray
                @test eltype(a) === eltype(b)
                @test a == b
            end
            test_ex(:([1 2 ; 3 4 ;;; 5 6 ; 7 8]))
            test_ex(:(Float64[1 2 ; 3 4 ;;; 5 6 ; 7 8]))
            test_ex(:([1 2 ;;; 3 4]))
            test_ex(:(Float32[1 2 ;;; 3 4]))
            test_ex(:([ 1 2
                        3 4
                        ;;;
                        5 6
                        7 8 ]))
            test_ex(:([1 ; 2 ;; 3 ; 4 ;;; 5 ; 6 ;; 7 ; 8]))
            test_ex(:([[[1 ; 2] ;; [3 ; 4]] ;;; [[5 ; 6] ;; [7 ; 8]]]))
        end
    end

    @testset "Methods" begin
        m = @MArray [11 13; 12 14]

        @test isimmutable(m) == false

        @test m[1] === 11
        @test m[2] === 12
        @test m[3] === 13
        @test m[4] === 14

        @testinf Tuple(m) === (11, 12, 13, 14)

        @test size(m) === (2, 2)
        @test size(typeof(m)) === (2, 2)
        @test size(MArray{Tuple{2,2},Int,2}) === (2, 2)
        @test size(MArray{Tuple{2,2},Int}) === (2, 2)
        @test size(MArray{Tuple{2,2}}) === (2, 2)

        @test size(m, 1) === 2
        @test size(m, 2) === 2
        @test size(typeof(m), 1) === 2
        @test size(typeof(m), 2) === 2

        @test length(m) === 4

        @test Base.mightalias(m, m)
        @test !Base.mightalias(m, copy(m))
        @test Base.mightalias(m, view(m, :, 1))

        @test Base.dataids(m) == (UInt(pointer(m)),)
    end

    @testset "setindex!" begin
        v = @MArray [1,2,3]
        v[1] = 11
        v[2] = 12
        v[3] = 13
        @test v.data === (11, 12, 13)
        @test setindex!(v, 11, 1) === v

        m = @MArray [0 0; 0 0]
        m[1] = 11
        m[2] = 12
        m[3] = 13
        m[4] = 14
        @test m.data === (11, 12, 13, 14)
        @test setindex!(m, 11, 1, 1) === m

        @test_throws BoundsError setindex!(v, 4, -1)
        mm = @MArray zeros(3,3,3,3)
        @test_throws BoundsError setindex!(mm, 4, -1)
        @test_throws BoundsError setindex!(mm, 4, 82)

        # setindex with non-elbits type
        m = MArray{Tuple{2,2,2}, String}(undef)
        @test_throws ErrorException setindex!(m, "a", 1, 1, 1)
    end

    @testset "rand! randn! randexp!" begin
        @test isa(rand!(@MArray zeros(2,2,1)), MArray{Tuple{2,2,1}, Float64})
        @test isa(randn!(@MArray zeros(2,2,1)), MArray{Tuple{2,2,1}, Float64})
        @test isa(randexp!(@MArray zeros(2,2,1)), MArray{Tuple{2,2,1}, Float64})
    end

    @testset "promotion" begin
        @test @inferred(promote_type(MVector{1,Float64}, MVector{1,BigFloat})) == MVector{1,BigFloat}
        @test @inferred(promote_type(MVector{2,Int}, MVector{2,Float64})) === MVector{2,Float64}
        @test @inferred(promote_type(MMatrix{2,3,Float32,6}, MMatrix{2,3,Complex{Float64},6})) === MMatrix{2,3,Complex{Float64},6}
        @test @inferred(promote_type(MArray{Tuple{2, 2, 2, 2},Float32, 4, 16}, MArray{Tuple{2, 2, 2, 2}, Complex{Float64}, 4, 16})) === MArray{Tuple{2, 2, 2, 2}, Complex{Float64}, 4, 16}
    end

    @testset "zero-dimensional" begin
        v = MArray{Tuple{}, Int, 0, 1}(1)
        @test v[] == 1
        v[] = 2
        @test v[] == 2
    end

    @testset "boolean indexing" begin
        v = @MArray [1,2,3]
        b = view(v, SA[true, false, true])
        @test b == [1,3]
    end
    
    @testset "non-power-of-2 element size" begin
        primitive type Test24 24 end
        Test24(n) = Base.trunc_int(Test24, n)
        a = Test24.(1:4)
        m = MVector{4}(a)
        @test m == m[:] == m[1:4] == a
        @test getindex.(Ref(m), 1:4) == a
        @test GC.@preserve m unsafe_load.(pointer(m), 1:4) == a
        @test GC.@preserve m unsafe_load.(pointer.(Ref(m), 1:4)) == a
        b = Test24.(5:8)
        setindex!.(Ref(m), b, 1:4)
        @test m == b
    end
end

@testset "some special case" begin
    @test_throws Exception MVector{1}(1, 2)
    @test (@inferred(MVector{1}((1, 2)))::MVector{1,NTuple{2,Int}}).data === ((1,2),)
    @test (@inferred(MVector{2}((1, 2)))::MVector{2,Int}).data === (1,2)
    @test (@inferred(MVector(1, 2))::MVector{2,Int}).data === (1,2)
    @test (@inferred(MVector((1, 2)))::MVector{2,Int}).data === (1,2)

    @test_throws Exception MMatrix{1,1}(1, 2)
    @test (@inferred(MMatrix{1,1}((1, 2)))::MMatrix{1,1,NTuple{2,Int}}).data === ((1,2),)
    @test (@inferred(MMatrix{1,2}((1, 2)))::MMatrix{1,2,Int}).data === (1,2)
    @test (@inferred(MMatrix{1}((1, 2)))::MMatrix{1,2,Int}).data === (1,2)
    @test (@inferred(MMatrix{1}(1, 2))::MMatrix{1,2,Int}).data === (1,2)
    @test (@inferred(MMatrix{2}((1, 2)))::MMatrix{2,1,Int}).data === (1,2)
    @test (@inferred(MMatrix{2}(1, 2))::MMatrix{2,1,Int}).data === (1,2)
end
