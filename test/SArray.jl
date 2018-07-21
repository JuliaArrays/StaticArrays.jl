@testset "SArray" begin
    @testset "Inner Constructors" begin
        @test SArray{Tuple{1},Int,1,1}((1,)).data === (1,)
        @test SArray{Tuple{1},Float64,1,1}((1,)).data === (1.0,)
        @test SArray{Tuple{2,2},Float64,2,4}((1, 1.0, 1, 1)).data === (1.0, 1.0, 1.0, 1.0)

        # Bad input
        @test_throws Exception SArray{Tuple{1},Int,1,1}()
        @test_throws Exception SArray{Tuple{2},Int,1,2}((1,))
        @test_throws Exception SArray{Tuple{1},Int,1,1}(())

        # Bad parameters
        @test_throws Exception SArray{Tuple{1},Int,1,2}((1,))
        @test_throws Exception SArray{Tuple{1},Int,2,1}((1,))
        @test_throws Exception SArray{Tuple{1},1,1,1}((1,))
        @test_throws Exception SArray{Tuple{2},Int,1,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        @test SArray{Tuple{1},Int,1}((1,)).data === (1,)
        @test SArray{Tuple{1},Int}((1,)).data === (1,)
        @test SArray{Tuple{1}}((1,)).data === (1,)

        @test SArray{Tuple{2,2},Int,2}((1,2,3,4)).data === (1,2,3,4)
        @test SArray{Tuple{2,2},Int}((1,2,3,4)).data === (1,2,3,4)
        @test SArray{Tuple{2,2}}((1,2,3,4)).data === (1,2,3,4)

        @test SArray(SArray{Tuple{2}}(1,2)) === SArray{Tuple{2}}(1,2)

        @test ((@SArray [1])::SArray{Tuple{1}}).data === (1,)
        @test ((@SArray [1,2])::SArray{Tuple{2}}).data === (1,2)
        @test ((@SArray Float64[1,2,3])::SArray{Tuple{3}}).data === (1.0, 2.0, 3.0)
        @test ((@SArray [1 2])::SArray{Tuple{1,2}}).data === (1, 2)
        @test ((@SArray Float64[1 2])::SArray{Tuple{1,2}}).data === (1.0, 2.0)
        @test ((@SArray [1 ; 2])::SArray{Tuple{2,1}}).data === (1, 2)
        @test ((@SArray Float64[1 ; 2])::SArray{Tuple{2,1}}).data === (1.0, 2.0)
        @test ((@SArray [1 2 ; 3 4])::SArray{Tuple{2,2}}).data === (1, 3, 2, 4)
        @test ((@SArray Float64[1 2 ; 3 4])::SArray{Tuple{2,2}}).data === (1.0, 3.0, 2.0, 4.0)

        @test ((@SArray [i for i = 1:2])::SArray{Tuple{2}}).data === (1, 2)
        @test ((@SArray [i*j for i = 1:2, j = 2:3])::SArray{Tuple{2,2}}).data === (2, 4, 3, 6)
        @test ((@SArray [i*j*k for i = 1:2, j = 2:3, k = 3:4])::SArray{Tuple{2,2,2}}).data === (6, 12, 9, 18, 8, 16, 12, 24)
        @test ((@SArray [i*j*k*l for i = 1:2, j = 2:3, k = 3:4, l = 1:2])::SArray{Tuple{2,2,2,2}}).data === (6, 12, 9, 18, 8, 16, 12, 24, 12, 24, 18, 36, 16, 32, 24, 48)
        @test ((@SArray [i*j*k*l*m for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2])::SArray{Tuple{2,2,2,2,2}}).data === (6, 12, 9, 18, 8, 16, 12, 24, 12, 24, 18, 36, 16, 32, 24, 48, 2*6, 2*12, 2*9, 2*18, 2*8, 2*16, 2*12, 2*24, 2*12, 2*24, 2*18, 2*36, 2*16, 2*32, 2*24, 2*48)
        @test ((@SArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2])::SArray{Tuple{2,2,2,2,2,2}}).data === ntuple(i->1, 64)
        @test ((@SArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2])::SArray{Tuple{2,2,2,2,2,2,2}}).data === ntuple(i->1, 128)
        @test ((@SArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2])::SArray{Tuple{2,2,2,2,2,2,2,2}}).data === ntuple(i->1, 256)
        test_expand_error(:(@SArray [1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2, q = 1:2]))
        @test ((@SArray Float64[i for i = 1:2])::SArray{Tuple{2}}).data === (1.0, 2.0)
        @test ((@SArray Float64[i*j for i = 1:2, j = 2:3])::SArray{Tuple{2,2}}).data === (2.0, 4.0, 3.0, 6.0)
        @test ((@SArray Float64[i*j*k for i = 1:2, j = 2:3, k =3:4])::SArray{Tuple{2,2,2}}).data === (6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0)
        @test ((@SArray Float64[i*j*k*l for i = 1:2, j = 2:3, k = 3:4, l = 1:2])::SArray{Tuple{2,2,2,2}}).data === (6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0, 12.0, 24.0, 18.0, 36.0, 16.0, 32.0, 24.0, 48.0)
        @test ((@SArray Float64[i*j*k*l*m for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2])::SArray{Tuple{2,2,2,2,2}}).data === (6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0, 12.0, 24.0, 18.0, 36.0, 16.0, 32.0, 24.0, 48.0, 2*6.0, 2*12.0, 2*9.0, 2*18.0, 2*8.0, 2*16.0, 2*12.0, 2*24.0, 2*12.0, 2*24.0, 2*18.0, 2*36.0, 2*16.0, 2*32.0, 2*24.0, 2*48.0)
        @test ((@SArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2])::SArray{Tuple{2,2,2,2,2,2}}).data === ntuple(i->1.0, 64)
        @test ((@SArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2])::SArray{Tuple{2,2,2,2,2,2,2}}).data === ntuple(i->1.0, 128)
        @test ((@SArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2])::SArray{Tuple{2,2,2,2,2,2,2,2}}).data === ntuple(i->1.0, 256)
        test_expand_error(:(@SArray Float64[1 for i = 1:2, j = 2:3, k = 3:4, l = 1:2, m = 1:2, n = 1:2, o = 1:2, p = 1:2, q = 1:2]))

        test_expand_error(:(@SArray [1 2; 3]))
        test_expand_error(:(@SArray Float64[1 2; 3]))
        test_expand_error(:(@SArray ones))
        test_expand_error(:(@SArray fill))
        test_expand_error(:(@SArray ones()))
        test_expand_error(:(@SArray sin(1:5)))
        test_expand_error(:(@SArray fill()))
        test_expand_error(:(@SArray fill(1)))
        test_expand_error(:(@SArray [1; 2; 3; 4]...))

        @test ((@SArray fill(3.,2,2,1))::SArray{Tuple{2,2,1}, Float64}).data === (3.0, 3.0, 3.0, 3.0)
        @test ((@SArray zeros(2,2,1))::SArray{Tuple{2,2,1}, Float64}).data === (0.0, 0.0, 0.0, 0.0)
        @test ((@SArray ones(2,2,1))::SArray{Tuple{2,2,1}, Float64}).data === (1.0, 1.0, 1.0, 1.0)
        @test isa(@SArray(rand(2,2,1)), SArray{Tuple{2,2,1}, Float64})
        @test isa(@SArray(randn(2,2,1)), SArray{Tuple{2,2,1}, Float64})
        @test isa(@SArray(randexp(2,2,1)), SArray{Tuple{2,2,1}, Float64})

        @test ((@SArray zeros(Float32, 2, 2, 1))::SArray{Tuple{2,2,1},Float32}).data === (0.0f0, 0.0f0, 0.0f0, 0.0f0)
        @test ((@SArray ones(Float32, 2, 2, 1))::SArray{Tuple{2,2,1},Float32}).data === (1.0f0, 1.0f0, 1.0f0, 1.0f0)
        @test isa(@SArray(rand(Float32, 2, 2, 1)), SArray{Tuple{2,2,1}, Float32})
        @test isa(@SArray(randn(Float32, 2, 2, 1)), SArray{Tuple{2,2,1}, Float32})
        @test isa(@SArray(randexp(Float32, 2, 2, 1)), SArray{Tuple{2,2,1}, Float32})

        m = [1 2; 3 4]
        @test SArray{Tuple{2,2}}(m) === @SArray [1 2; 3 4]

        # Non-square comprehensions built from SVectors - see #76
        @test @SArray([1 for x = SVector(1,2), y = SVector(1,2,3)]) == ones(2,3)
    end

    @testset "Methods" begin
        m = @SArray [11 13; 12 14]

        @test isimmutable(m) == true

        @test m[1] === 11
        @test m[2] === 12
        @test m[3] === 13
        @test m[4] === 14

        @testinf Tuple(m) === (11, 12, 13, 14)

        @test size(m) === (2, 2)
        @test size(typeof(m)) === (2, 2)
        @test size(SArray{Tuple{2,2},Int,2}) === (2, 2)
        @test size(SArray{Tuple{2,2},Int}) === (2, 2)
        @test size(SArray{Tuple{2,2}}) === (2, 2)

        @test size(m, 1) === 2
        @test size(m, 2) === 2
        @test size(typeof(m), 1) === 2
        @test size(typeof(m), 2) === 2

        @test length(m) === 4

        @test_throws Exception m[1] = 1

        if isdefined(Base, :dataids) # v0.7-
            @test Base.dataids(m) === ()
        end
    end

    @testset "promotion" begin
        @test @inferred(promote_type(SVector{1,Float64}, SVector{1,BigFloat})) == SVector{1,BigFloat}
        @test @inferred(promote_type(SVector{2,Int}, SVector{2,Float64})) === SVector{2,Float64}
        @test @inferred(promote_type(SMatrix{2,3,Float32,6}, SMatrix{2,3,Complex{Float64},6})) === SMatrix{2,3,Complex{Float64},6}
    end
end
