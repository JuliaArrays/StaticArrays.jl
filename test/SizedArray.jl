@testset "SizedArray" begin
    @testset "Inner Constructors" begin
        @test SizedArray{Tuple{2}, Int, 1}((3, 4)).data == [3, 4]
        @test SizedArray{Tuple{2}, Int, 1}([3, 4]).data == [3, 4]
        @test SizedArray{Tuple{2, 2}, Int, 2}(collect(3:6)).data == collect(3:6)
        @test size(SizedArray{Tuple{4, 5}, Int, 2}().data) == (4, 5)
        @test size(SizedArray{Tuple{4, 5}, Int}().data) == (4, 5)

        # Bad input
        @test_throws Exception SArray{Tuple{1},Int,1}([2 3])

        # Bad parameters
        @test_throws Exception SizedArray{Tuple{1},Int,2}()
        @test_throws Exception SArray{Tuple{3, 4},Int,1}()
        
        # Parameter/input size mismatch
        @test_throws Exception SizedArray{Tuple{1},Int,2}([2; 3])
        @test_throws Exception SizedArray{Tuple{1},Int,2}((2, 3))
    end

    # setindex
    sa = SizedArray{Tuple{2}, Int, 1}([3, 4])
    sa[1] = 2
    @test sa.data == [2, 4]

    @testset "back to Array" begin
        @test Array(SizedArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int}(SizedArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Array{Int, 1}(SizedArray{Tuple{2}, Int, 1}([3, 4])) == [3, 4]
        @test Vector(SizedArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test convert(Vector, SizedArray{Tuple{4}, Int, 1}(collect(3:6))) == collect(3:6)
        @test Matrix(SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
        @test convert(Matrix, SMatrix{2,2}((1,2,3,4))) == [1 3; 2 4]
    end
end
