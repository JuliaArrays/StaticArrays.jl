@testset "SDiagonal" begin
    @testset "Constructors" begin
        @test SDiagonal{1,Int64}((1,)).diag === SVector{1,Int64}((1,))
        @test SDiagonal{1,Float64}((1,)).diag === SVector{1,Float64}((1,))
     
        @test SDiagonal{4,Float64}((1, 1.0, 1, 1)).diag.data === (1.0, 1.0, 1.0, 1.0)

        # Bad input
        @test_throws Exception SMatrix{1,Int}()
        @test_throws Exception SMatrix{2,Int}((1,))

        # From SMatrix
        @test SDiagonal(SMatrix{2,2,Int}((1,2,3,4))).diag.data === (1,4)

    end

    @testset "Methods" begin
    
        @test StaticArrays.scalem(@SMatrix([1 1 1;1 1 1; 1 1 1]), @SVector [1,2,3]) === @SArray [1 2 3; 1 2 3; 1 2 3]
    
        m = SDiagonal(@SVector [11, 12, 13, 14])
        m2 = diagm([11, 12, 13, 14])
        
        b = @SVector [2,-1,2,1]
        b2 = Vector(b)
        
        @test isimmutable(m) == true

        @test m[1,1] === 11
        @test m[2,2] === 12
        @test m[3,3] === 13
        @test m[4,4] === 14
        
        for i in 1:4
            for j in 1:4
                i == j || @test m[i,j] === 0
            end
        end
        
        @test_throws Exception m[5,5]
        
        @test_throws Exception m[1,5]
        
    
        @test size(m) === (4, 4)
        @test size(typeof(m)) === (4, 4)
        @test size(SDiagonal{4}) === (4, 4)

        @test size(m, 1) === 4
        @test size(m, 2) === 4
        @test size(typeof(m), 1) === 4
        @test size(typeof(m), 2) === 4

        @test length(m) === 4*4

        @test_throws Exception m[1] = 1
        
        @test m*b ==  @SVector [22,-12,26,14]
        @test (b'*m)' ==  @SVector [22,-12,26,14]
        
        @test m\b == m2\b
        @test m*m == m2*m
        
        @test ishermitian(m) == ishermitian(m2)
        @test isposdef(m) == isposdef(m2)
        @test issymmetric(m) == issymmetric(m2)
        
        @test m' == m
        @test 2m == m + m
        @test 0m == m - m
        
        @test m\m == eye(SDiagonal{4,Float64})
        
        
        
        
    end
end