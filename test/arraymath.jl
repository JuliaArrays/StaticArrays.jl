@testset "Array math" begin
    @testset "Array-scalar math" begin
        m = @SMatrix [1 2; 3 4]

        @test m .+ 1 === @SMatrix [2 3; 4 5]
        @test 1 .+ m === @SMatrix [2 3; 4 5]
        @test m .* 2 === @SMatrix [2 4; 6 8]
        @test 2 .* m === @SMatrix [2 4; 6 8]
        @test m .- 1 === @SMatrix [0 1; 2 3]
        @test 1 .- m === @SMatrix [0 -1; -2 -3]
        @test m ./ 2 === @SMatrix [0.5 1.0; 1.5 2.0]
        @test 12 ./ m === @SMatrix [12.0 6.0; 4.0 3.0]

    end

    @testset "Elementwise array math" begin
        m1 = @SMatrix [1 2; 3 4]
        m2 = @SMatrix [4 3; 2 1]

        @test .-(m1) === @SMatrix [-1 -2; -3 -4]

        @test m1 .+ m2 === @SMatrix [5 5; 5 5]
        @test m1 .* m2 === @SMatrix [4 6; 6 4]
        @test m1 .- m2 === @SMatrix [-3 -1; 1 3]
        @test m1 ./ m2 === @SMatrix [0.25 2/3; 1.5 4.0]
    end
end
