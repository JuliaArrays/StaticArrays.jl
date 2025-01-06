@testset "SVector" begin
    @testset "Inner Constructors" begin
        @test SVector{1,Int}((1,)).data === (1,)
        @test SVector{1,Float64}((1,)).data === (1.0,)
        @test SVector{2,Float64}((1, 1.0)).data === (1.0, 1.0)

        @test_throws Exception SVector{1,Int}()
        @test_throws Exception SVector{2,Int}((1,))
        @test_throws Exception SVector{1,Int}(())
        @test_throws Exception SVector{Int,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        @test SVector{1}((1,)).data === (1,)
        @test SVector{1}((1.0,)).data === (1.0,)
        @test SVector((1,)).data === (1,)
        @test SVector((1.0,)).data === (1.0,)

        # test for #557-like issues
        @test (@inferred SVector(MVector{0,Float64}()))::SVector{0,Float64} == SVector{0,Float64}()

        @test SVector{3}(i for i in 1:3).data === (1,2,3)
        @test SVector{3}(float(i) for i in 1:3).data === (1.0,2.0,3.0)
        @test SVector{0,Int}().data === ()
        @test SVector{3,Int}(i for i in 1:3).data === (1,2,3)
        @test SVector{3,Float64}(i for i in 1:3).data === (1.0,2.0,3.0)
        @test SVector{1}(SVector(SVector(1.0), SVector(2.0))[j] for j in 1:1) == SVector((SVector(1.0),))
        @test_throws Exception SVector{3}(i for i in 1:2)
        @test_throws Exception SVector{3}(i for i in 1:4)
        @test_throws Exception SVector{3,Int}(i for i in 1:2)
        @test_throws Exception SVector{3,Int}(i for i in 1:4)

        @test SVector(1).data === (1,)
        @test SVector(1,1.0).data === (1.0,1.0)

        @test ((@SVector [1.0])::SVector{1}).data === (1.0,)
        @test ((@SVector [1, 2, 3])::SVector{3}).data === (1, 2, 3)
        @test ((@SVector Float64[1,2,3])::SVector{3}).data === (1.0, 2.0, 3.0)
        @test ((@SVector [i for i = 1:3])::SVector{3}).data === (1, 2, 3)
        @test ((@SVector Float64[i for i = 1:3])::SVector{3}).data === (1.0, 2.0, 3.0)

        @test ((@SVector zeros(2))::SVector{2, Float64}).data === (0.0, 0.0)
        @test ((@SVector ones(2))::SVector{2, Float64}).data === (1.0, 1.0)
        @test ((@SVector fill(2.5, 2))::SVector{2,Float64}).data === (2.5, 2.5)
        @test ((@SVector zeros(Float32, 2))::SVector{2,Float32}).data === (0.0f0, 0.0f0)
        @test ((@SVector ones(Float32, 2))::SVector{2,Float32}).data === (1.0f0, 1.0f0)

        @testset "@SVector rand*" begin
            n = 4
            @test (@SVector rand(n)) isa SVector{n, Float64}
            @test (@SVector randn(n)) isa SVector{n, Float64}
            @test (@SVector randexp(n)) isa SVector{n, Float64}
            @test (@SVector rand(4)) isa SVector{4, Float64}
            @test (@SVector randn(4)) isa SVector{4, Float64}
            @test (@SVector randexp(4)) isa SVector{4, Float64}
            @test (@SVector rand(_rng(), n)) isa SVector{n, Float64}
            @test (@SVector rand(_rng(), n)) == rand(_rng(), n)
            @test (@SVector randn(_rng(), n)) isa SVector{n, Float64}
            @test (@SVector randn(_rng(), n)) == randn(_rng(), n)
            @test (@SVector randexp(_rng(), n)) isa SVector{n, Float64}
            @test (@SVector randexp(_rng(), n)) == randexp(_rng(), n)
            @test (@SVector rand(_rng(), 4)) isa SVector{4, Float64}
            @test (@SVector rand(_rng(), 4)) == rand(_rng(), 4)
            @test (@SVector randn(_rng(), 4)) isa SVector{4, Float64}
            @test (@SVector randn(_rng(), 4)) == randn(_rng(), 4)
            @test (@SVector randexp(_rng(), 4)) isa SVector{4, Float64}
            @test (@SVector randexp(_rng(), 4)) == randexp(_rng(), 4)

            for T in (Float32, Float64)
                @test (@SVector rand(T, n)) isa SVector{n, T}
                @test (@SVector randn(T, n)) isa SVector{n, T}
                @test (@SVector randexp(T, n)) isa SVector{n, T}
                @test (@SVector rand(T, 4)) isa SVector{4, T}
                @test (@SVector randn(T, 4)) isa SVector{4, T}
                @test (@SVector randexp(T, 4)) isa SVector{4, T}
                @test (@SVector rand(_rng(), T, n)) isa SVector{n, T}
                VERSION≥v"1.7" && @test (@SVector rand(_rng(), T, n)) == rand(_rng(), T, n) broken=(T===Float32)
                @test (@SVector randn(_rng(), T, n)) isa SVector{n, T}
                @test (@SVector randn(_rng(), T, n)) == randn(_rng(), T, n)
                @test (@SVector randexp(_rng(), T, n)) isa SVector{n, T}
                @test (@SVector randexp(_rng(), T, n)) == randexp(_rng(), T, n)
                @test (@SVector rand(_rng(), T, 4)) isa SVector{4, T}
                VERSION≥v"1.7" && @test (@SVector rand(_rng(), T, 4)) == rand(_rng(), T, 4) broken=(T===Float32)
                @test (@SVector randn(_rng(), T, 4)) isa SVector{4, T}
                @test (@SVector randn(_rng(), T, 4)) == randn(_rng(), T, 4)
                @test (@SVector randexp(_rng(), T, 4)) isa SVector{4, T}
                @test (@SVector randexp(_rng(), T, 4)) == randexp(_rng(), T, 4)
            end
        end

        @testset "expand error" begin
            test_expand_error(:(@SVector fill(1.5, 2, 3)))
            test_expand_error(:(@SVector ones(2, 3, 4)))
            test_expand_error(:(@SVector rand(Float64, 2, 3, 4)))
            test_expand_error(:(@SVector sin(1:5)))
            test_expand_error(:(@SVector [i*j for i in 1:2, j in 2:3]))
            test_expand_error(:(@SVector Float32[i*j for i in 1:2, j in 2:3]))
            test_expand_error(:(@SVector [1; 2; 3]...))
            test_expand_error(:(@SVector a))
            test_expand_error(:(@SVector [[1 2];[3 4]]))
        end

        if VERSION >= v"1.7.0"
            @test ((@SVector Float64[1;2;3;;;])::SVector{3}).data === (1.0, 2.0, 3.0)
            @test ((@SVector [1;2;3;;;])::SVector{3}).data === (1, 2, 3)
        end
    end

    @testset "Methods" begin
        v = @SVector [11, 12, 13]

        @test isimmutable(v) == true

        @test v[1] === 11
        @test v[2] === 12
        @test v[3] === 13

        @testinf Tuple(v) === (11, 12, 13)

        @test size(v) === (3,)
        @test size(typeof(v)) === (3,)
        @test size(SVector{3}) === (3,)

        @test size(v, 1) === 3
        @test size(v, 2) === 1
        @test size(typeof(v), 1) === 3
        @test size(typeof(v), 2) === 1

        @test length(v) === 3

        @test_throws Exception v[1] = 1

        @test (@inferred view(v, :)) === v
        @test (@inferred view(v, 1)) === Scalar(v[1])
        @test (@inferred view(v, CartesianIndex(1))) === Scalar(v[1])
        @test (@inferred view(v, CartesianIndex(1,1))) === Scalar(v[1])
        @test (@inferred view(v, 1, CartesianIndex(1))) === Scalar(v[1])
        @test (@inferred view(v, SVector{2,Int}(1,2))) === @SArray [11, 12]
        @test (@inferred view(v, SOneTo(2))) === @SArray [11, 12]

        @test reverse(v) == reverse(collect(v), dims = 1)
    end

    @testset "CartesianIndex" begin
        a, b = SVector(0x01, 0x02), SVector(1.0f0, 1.2f0)
        c = CartesianIndex((1,2))
        @test @inferred(eltype([a,c])) == SVector{2,Int}
        @test @inferred(eltype([b,c])) == SVector{2,Float32}
        @test @inferred(convert(SVector, c)) == SVector{2,Int}([1, 2])
        @test @inferred(convert(SVector{2}, c)) == SVector{2,Int}([1, 2])
    end

    @testset "Named field access - getproperty" begin
        @test propertynames(SA[1,2,3,4,5]) == ()
        @test propertynames(SA[1,2,3,4,5], true) == (:data,)
        @test propertynames(SA[1 2; 3 4], true) == (:data,)
        v4 = SA[10,20,30,40]
        @test propertynames(v4) == (:x, :y, :z, :w)
        @test propertynames(v4, true) == (:x, :y, :z, :w, :data)
        @test v4.x == 10
        @test v4.y == 20
        @test v4.z == 30
        @test v4.w == 40
        v2 = SA[10,20]
        @test propertynames(v2) == (:x, :y)
        @test propertynames(v2, true) == (:x, :y, :data)
        @test v2.x == 10
        @test v2.y == 20
        @test_throws ErrorException v2.z
        @test_throws ErrorException v2.w
    end

    @testset "issue 1042" begin
        f = [1,2,3]
        @test f == @SVector [f[i] for i in 1:3]
    end

    @testset "issue 1118" begin
        a = SVector{1}(1)
        @test SVector{1, Tuple{SVector{1, Int}, SVector{1, Int}}}((a,a)) === SVector{1}((a,a))
        @test SVector{1, NTuple}((a,a))[1] === (a,a)
    end
end
