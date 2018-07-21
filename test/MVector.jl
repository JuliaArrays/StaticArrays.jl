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

        @test ((@MVector [1.0])::MVector{1}).data === (1.0,)
        @test ((@MVector [1, 2, 3])::MVector{3}).data === (1, 2, 3)
        @test ((@MVector Float64[1,2,3])::MVector{3}).data === (1.0, 2.0, 3.0)
        @test ((@MVector [i for i = 1:3])::MVector{3}).data === (1, 2, 3)
        @test ((@MVector Float64[i for i = 1:3])::MVector{3}).data === (1.0, 2.0, 3.0)

        @test ((@MVector zeros(2))::MVector{2, Float64}).data === (0.0, 0.0)
        @test ((@MVector ones(2))::MVector{2, Float64}).data === (1.0, 1.0)
        @test ((@MVector fill(2.5, 2))::MVector{2, Float64}).data === (2.5, 2.5)
        @test isa(@MVector(rand(2)), MVector{2, Float64})
        @test isa(@MVector(randn(2)), MVector{2, Float64})
        @test isa(@MVector(randexp(2)), MVector{2, Float64})

        @test ((@MVector zeros(Float32, 2))::MVector{2,Float32}).data === (0.0f0, 0.0f0)
        @test ((@MVector ones(Float32, 2))::MVector{2,Float32}).data === (1.0f0, 1.0f0)
        @test isa(@MVector(rand(Float32, 2)), MVector{2, Float32})
        @test isa(@MVector(randn(Float32, 2)), MVector{2, Float32})
        @test isa(@MVector(randexp(Float32, 2)), MVector{2, Float32})

        test_expand_error(:(@MVector fill(1.5, 2, 3)))
        test_expand_error(:(@MVector ones(2, 3, 4)))
        test_expand_error(:(@MVector sin(1:5)))
        test_expand_error(:(@MVector [i*j for i in 1:2, j in 2:3]))
        test_expand_error(:(@MVector Float32[i*j for i in 1:2, j in 2:3]))
        test_expand_error(:(@MVector [1; 2; 3]...))
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
    end

    @testset "setindex!" begin
        v = @MVector [1,2,3]
        v[1] = 11
        v[2] = 12
        v[3] = 13
        @test v.data === (11, 12, 13)

        v = @MVector [1.,2.,3.]
        v[1] = Float16(11)
        @test v.data === (11., 2., 3.)

        @test_throws BoundsError setindex!(v, 4., -1)
        @test_throws BoundsError setindex!(v, 4., 4)

        # setindex with non-elbits type
        v = MVector{2,String}(undef)
        @test_throws ErrorException setindex!(v, "a", 1)
    end
end
