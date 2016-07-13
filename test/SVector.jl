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

        @test (@SVector [1.0]).data === (1.0,)
        @test (@SVector [1, 2, 3]).data === (1, 2, 3)
    end

    @testset "Methods" begin
        v = @SVector [11, 12, 13]

        @test isimmutable(v) == true

        @test v[1] === 11
        @test v[2] === 12
        @test v[3] === 13

        @test Tuple(v) === (11, 12, 13)

        @test size(v) === (3,)
        @test size(typeof(v)) === (3,)
        @test size(SVector{3}) === (3,)

        @test size(v, 1) === 3
        @test size(v, 2) === 1
        @test size(typeof(v), 1) === 3
        @test size(typeof(v), 2) === 1

        @test length(v) === 3

        @test_throws Exception v[1] = 1
    end
end
