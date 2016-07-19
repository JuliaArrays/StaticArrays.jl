@testset "Matrix multiplication" begin
    @testset "Matrix-vector" begin
        m = @SMatrix [1 2; 3 4]
        v = @SVector [1, 2]
        @test m*v === @SVector [5, 11]

        m = @MMatrix [1 2; 3 4]
        v = @MVector [1, 2]
        @test (m*v)::MVector == @MVector [5, 11]

        m = @SArray [1 2; 3 4]
        v = @SArray [1, 2]
        @test m*v === @SArray [5, 11]

        m = @MArray [1 2; 3 4]
        v = @MArray [1, 2]
        @test (m*v)::MArray == @MArray [5, 11]
    end

    @testset "Matrix-matrix" begin
        m = @SMatrix [1 2; 3 4]
        n = @SMatrix [2 3; 4 5]
        @test m*n === @SMatrix [10 13; 22 29]

        m = @MMatrix [1 2; 3 4]
        n = @MMatrix [2 3; 4 5]
        @test (m*n)::MMatrix == @MMatrix [10 13; 22 29]

        m = @SArray [1 2; 3 4]
        n = @SArray [2 3; 4 5]
        @test m*n === @SArray [10 13; 22 29]

        m = @MArray [1 2; 3 4]
        n = @MArray [2 3; 4 5]
        @test (m*n)::MArray == @MArray [10 13; 22 29] # TODO maybe make these remember their type
    end

    @testset "A_mul_B!" begin
        m = @SMatrix [1 2; 3 4]
        n = @SMatrix [2 3; 4 5]

        a = MMatrix{2,2,Int,4}()
        A_mul_B!(a, m, n)
        @test a::MMatrix{2,2,Int,4} == @MMatrix [10 13; 22 29]

        a2 = MArray{(2,2),Int,2,4}()
        A_mul_B!(a2, m, n)
        @test a2::MArray{(2,2),Int,2,4} == @MArray [10 13; 22 29]
    end
end
