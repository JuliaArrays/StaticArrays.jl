@testset "MVector" begin
    @testset "Inner Constructors" begin
        @test MVector{1,Int}((1,)).data === (1,)
        @test MVector{1,Float64}((1,)).data === (1.0,)
        @test MVector{2,Float64}((1,1.0)).data === (1.0,1.0)
        @test isa(MVector{1,Int}(), MVector{1,Int})

        @test_throws Exception MVector{2,Int}((1,))
        @test_throws Exception MVector{1,Int}(())
        @test_throws Exception MVector{Int,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        @test MVector{1}((1,)).data === (1,)
        @test MVector{1}((1.0,)).data === (1.0,)
        @test MVector((1,)).data === (1,)
        @test MVector((1.0,)).data === (1.0,)

        @test ((@MVector [1.0])::MVector{1}).data === (1.0,)
        @test ((@MVector [1, 2, 3])::MVector{3}).data === (1, 2, 3)
        @test ((@MVector Float64[1,2,3])::MVector{3}).data === (1.0, 2.0, 3.0)
    end

    @testset "Methods" begin
        v = @MVector [11, 12, 13]

        @test isimmutable(v) == false

        @test v[1] === 11
        @test v[2] === 12
        @test v[3] === 13

        @test Tuple(v) === (11, 12, 13)

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
    end
end
