@testset "SizedArray" begin
    @testset "Inner Constructors" begin
        @test SizedArray{Tuple{2}, Int, 1}([3, 4]).data == [3, 4]
        @test SizedArray{Tuple{2, 2}, Int, 2}(collect(3:6)).data == collect(3:6)
        @test size(SizedArray{Tuple{4, 5}, Int, 2}().data) == (4, 5)
        @test size(SizedArray{Tuple{4, 5}, Int}().data) == (4, 5)

        # Bad input
        @test_throws Exception SArray{Tuple{1},Int,1}([2 3])

        # Bad parameters
        @test_throws Exception SizedArray{Tuple{1},Int,2}()
        @test_throws Exception SArray{Tuple{3, 4},Int,1}()
    end
end
