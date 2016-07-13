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
        @test_throws Exception SMatrix{1,1,1,1}((1,))
        @test_throws Exception SMatrix{1,2,Int,1}((1,))
        @test_throws Exception SMatrix{2,1,Int,1}((1,))
    end

    @testset "Outer constructors and macro" begin
        @test SMatrix{1,1,Int}((1,)).data === (1,)
        @test SMatrix{1,1}((1,)).data === (1,)
        @test SMatrix{1}((1,)).data === (1,)

        @test SMatrix{2,2,Int}((1,2,3,4)).data === (1,2,3,4)
        @test SMatrix{2,2}((1,2,3,4)).data === (1,2,3,4)
        @test SMatrix{2}((1,2,3,4)).data === (1,2,3,4)

        @test ((@SMatrix [1.0])::SMatrix{1,1}).data === (1.0,)
        @test ((@SMatrix [1 2])::SMatrix{1,2}).data === (1, 2)
        @test ((@SMatrix [1 ; 2])::SMatrix{2,1}).data === (1, 2)
        @test ((@SMatrix [1 2 ; 3 4])::SMatrix{2,2}).data === (1, 3, 2, 4)

        @test (ex = macroexpand(:(@SMatrix [1 2; 3])); isa(ex, Expr) && ex.head == :error)
    end

    @testset "Methods" begin
        m = @SMatrix [11 13; 12 14]

        @test isimmutable(m) == true

        @test m[1] === 11
        @test m[2] === 12
        @test m[3] === 13
        @test m[4] === 14

        @test Tuple(m) === (11, 12, 13, 14)

        @test size(m) === (2, 2)
        @test size(typeof(m)) === (2, 2)
        @test size(SMatrix{2,2}) === (2, 2)

        @test size(m, 1) === 2
        @test size(m, 2) === 2
        @test size(typeof(m), 1) === 2
        @test size(typeof(m), 2) === 2

        @test length(m) === 4

        @test_throws Exception m[1] = 1
    end
end
