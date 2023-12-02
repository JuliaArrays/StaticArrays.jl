@testset "MVector" begin
    @testset "Inner Constructors" begin
        @test MVector{1,Int}((1,)).data === (1,)
        @test MVector{1,Float64}((1,)).data === (1.0,)
        @test MVector{2,Float64}((1,1.0)).data === (1.0,1.0)
        @test isa(MVector{1,Int}(undef), MVector{1,Int})

        @test_throws Exception MVector{2,Int}((1,))
        @test_throws Exception MVector{1,Int}(())
        @test_throws Exception MVector{Int,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        @test MVector{1}((1,)).data === (1,)
        @test MVector{1}((1.0,)).data === (1.0,)
        @test MVector((1,)).data === (1,)
        @test MVector((1.0,)).data === (1.0,)

        # Constructors should create a copy (#335)
        v = MVector(1,2)
        @test MVector(v) !== v && MVector(v) == v

        # test for #557-like issues
        @test (@inferred MVector(SVector{0,Float64}()))::MVector{0,Float64} == MVector{0,Float64}()

        @test MVector{3}(i for i in 1:3).data === (1,2,3)
        @test MVector{3}(float(i) for i in 1:3).data === (1.0,2.0,3.0)
        @test MVector{0,Int}().data === ()
        @test MVector{3,Int}(i for i in 1:3).data === (1,2,3)
        @test MVector{3,Float64}(i for i in 1:3).data === (1.0,2.0,3.0)
        @test MVector{1}(MVector(MVector(1.0), MVector(2.0))[j] for j in 1:1) == MVector((MVector(1.0),))
        @test_throws Exception MVector{3}(i for i in 1:2)
        @test_throws Exception MVector{3}(i for i in 1:4)
        @test_throws Exception MVector{3,Int}(i for i in 1:2)
        @test_throws Exception MVector{3,Int}(i for i in 1:4)

        @test MVector(1).data === (1,)
        @test MVector(1,1.0).data === (1.0,1.0)

        @test ((@MVector [1.0])::MVector{1}).data === (1.0,)
        @test ((@MVector [1, 2, 3])::MVector{3}).data === (1, 2, 3)
        @test ((@MVector Float64[1,2,3])::MVector{3}).data === (1.0, 2.0, 3.0)
        @test ((@MVector [i for i = 1:3])::MVector{3}).data === (1, 2, 3)
        @test ((@MVector Float64[i for i = 1:3])::MVector{3}).data === (1.0, 2.0, 3.0)

        @test ((@MVector zeros(2))::MVector{2, Float64}).data === (0.0, 0.0)
        @test ((@MVector ones(2))::MVector{2, Float64}).data === (1.0, 1.0)
        @test ((@MVector fill(2.5, 2))::MVector{2,Float64}).data === (2.5, 2.5)
        @test ((@MVector zeros(Float32, 2))::MVector{2,Float32}).data === (0.0f0, 0.0f0)
        @test ((@MVector ones(Float32, 2))::MVector{2,Float32}).data === (1.0f0, 1.0f0)

        @testset "@MVector rand*" begin
            n = 4
            @test (@MVector rand(n)) isa MVector{n, Float64}
            @test (@MVector randn(n)) isa MVector{n, Float64}
            @test (@MVector randexp(n)) isa MVector{n, Float64}
            @test (@MVector rand(4)) isa MVector{4, Float64}
            @test (@MVector randn(4)) isa MVector{4, Float64}
            @test (@MVector randexp(4)) isa MVector{4, Float64}
            @test (@MVector rand(_rng(), n)) isa MVector{n, Float64}
            @test (@MVector rand(_rng(), n)) == rand(_rng(), n)
            @test (@MVector randn(_rng(), n)) isa MVector{n, Float64}
            @test (@MVector randn(_rng(), n)) == randn(_rng(), n)
            @test (@MVector randexp(_rng(), n)) isa MVector{n, Float64}
            @test (@MVector randexp(_rng(), n)) == randexp(_rng(), n)
            @test (@MVector rand(_rng(), 4)) isa MVector{4, Float64}
            @test (@MVector rand(_rng(), 4)) == rand(_rng(), 4)
            @test (@MVector randn(_rng(), 4)) isa MVector{4, Float64}
            @test (@MVector randn(_rng(), 4)) == randn(_rng(), 4)
            @test (@MVector randexp(_rng(), 4)) isa MVector{4, Float64}
            @test (@MVector randexp(_rng(), 4)) == randexp(_rng(), 4)

            for T in (Float32, Float64)
                @test (@MVector rand(T, n)) isa MVector{n, T}
                @test (@MVector randn(T, n)) isa MVector{n, T}
                @test (@MVector randexp(T, n)) isa MVector{n, T}
                @test (@MVector rand(T, 4)) isa MVector{4, T}
                @test (@MVector randn(T, 4)) isa MVector{4, T}
                @test (@MVector randexp(T, 4)) isa MVector{4, T}
                @test (@MVector rand(_rng(), T, n)) isa MVector{n, T}
                VERSION≥v"1.7" && @test (@MVector rand(_rng(), T, n)) == rand(_rng(), T, n) broken=(T===Float32)
                @test (@MVector randn(_rng(), T, n)) isa MVector{n, T}
                @test (@MVector randn(_rng(), T, n)) == randn(_rng(), T, n)
                @test (@MVector randexp(_rng(), T, n)) isa MVector{n, T}
                @test (@MVector randexp(_rng(), T, n)) == randexp(_rng(), T, n)
                @test (@MVector rand(_rng(), T, 4)) isa MVector{4, T}
                VERSION≥v"1.7" && @test (@MVector rand(_rng(), T, 4)) == rand(_rng(), T, 4) broken=(T===Float32)
                @test (@MVector randn(_rng(), T, 4)) isa MVector{4, T}
                @test (@MVector randn(_rng(), T, 4)) == randn(_rng(), T, 4)
                @test (@MVector randexp(_rng(), T, 4)) isa MVector{4, T}
                @test (@MVector randexp(_rng(), T, 4)) == randexp(_rng(), T, 4)
            end
        end

        @testset "expand error" begin
            test_expand_error(:(@MVector fill(1.5, 2, 3)))
            test_expand_error(:(@MVector ones(2, 3, 4)))
            test_expand_error(:(@MVector rand(Float64, 2, 3, 4)))
            test_expand_error(:(@MVector sin(1:5)))
            test_expand_error(:(@MVector [i*j for i in 1:2, j in 2:3]))
            test_expand_error(:(@MVector Float32[i*j for i in 1:2, j in 2:3]))
            test_expand_error(:(@MVector [1; 2; 3]...))
            test_expand_error(:(@MVector a))
            test_expand_error(:(@MVector [[1 2];[3 4]]))
        end

        if VERSION >= v"1.7.0"
            @test ((@MVector Float64[1;2;3;;;])::MVector{3}).data === (1.0, 2.0, 3.0)
            @test ((@MVector [1;2;3;;;])::MVector{3}).data === (1, 2, 3)
        end
    end

    @testset "Methods" begin
        v = @MVector [11, 12, 13]

        @test isimmutable(v) == false

        @test v[1] === 11
        @test v[2] === 12
        @test v[3] === 13

        @testinf Tuple(v) === (11, 12, 13)

        @test size(v) === (3,)
        @test size(typeof(v)) === (3,)
        @test size(MVector{3}) === (3,)

        @test size(v, 1) === 3
        @test size(v, 2) === 1
        @test size(typeof(v), 1) === 3
        @test size(typeof(v), 2) === 1

        @test length(v) === 3

        @test reverse(v) == reverse(collect(v), dims = 1)
    end

    @testset "setindex!" begin
        v = @MVector [1,2,3]
        v[1] = 11
        v[2] = 12
        v[3] = 13
        @test v.data === (11, 12, 13)
        @test setindex!(v, 13, 3) === v

        v = @MVector [1.,2.,3.]
        v[1] = Float16(11)
        @test v.data === (11., 2., 3.)

        @test_throws BoundsError setindex!(v, 4., -1)
        @test_throws BoundsError setindex!(v, 4., 4)

        # setindex with non-elbits type
        v = MVector{2,String}(undef)
        @test_throws ErrorException setindex!(v, "a", 1)
    end

    @testset "Named field access - getproperty/setproperty!" begin
        # getproperty
        v4 = @MVector [10,20,30,40]
        @test v4.x == 10
        @test v4.y == 20
        @test v4.z == 30
        @test v4.w == 40

        v2 = @MVector [10,20]
        @test v2.x == 10
        @test v2.y == 20
        @test_throws ErrorException v2.z
        @test_throws ErrorException v2.w

        # setproperty!
        @test (v4.x = 100) == 100
        @test (v4.y = 200) == 200
        @test (v4.z = 300) == 300
        @test (v4.w = 400) == 400
        @test v4[1] == 100
        @test v4[2] == 200
        @test v4[3] == 300
        @test v4[4] == 400

        @test (v2.x = 100) == 100
        @test (v2.y = 200) == 200
        @test_throws ErrorException (v2.z = 200)
        @test_throws ErrorException (v2.w = 200)
    end
end
