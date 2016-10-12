@testset "Scalar" begin
    @test Scalar(2) .* [1, 2, 3] == [2, 4, 6]
    @test Scalar([1 2; 3 4]) .+ [[1 1; 1 1], [2 2; 2 2]] == [[2 3; 4 5], [3 4; 5 6]]
end
