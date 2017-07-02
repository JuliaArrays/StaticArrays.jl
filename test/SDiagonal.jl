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
        m = SDiagonal(@SVector [11, 12, 13, 14])

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
    end
end