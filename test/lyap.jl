using StaticArrays, Test, LinearAlgebra

@testset "Lyapunov equation" begin
    @test lyap(@SMatrix([1]), @SMatrix [2]) === @SMatrix [-1.0]

    @test lyap(@SMatrix([1 1; 0 1]), @SMatrix [2 1; 1 2]) == [-1 0; 0 -1]
    @test isa(lyap(@SMatrix([1 1; 0 1]), @SMatrix [2 1; 1 2]), SArray{Tuple{2,2},Float64,2,4})

    @test lyap(@SMatrix([1 0 1; 0 1 0; 0 0 1]), @SMatrix [2 4 4; 0 4 20; 0 0 20]) == [-5.0 -2.0 3.0; 5.0 -2.0 -10.0; 5.0 0.0 -10.0]
    @test isa(lyap(@SMatrix([1 0 1; 0 1 0; 0 0 1]), @SMatrix [2 4 4; 0 4 20; 0 0 20]), SizedArray{Tuple{3,3},Float64,2,2})
end
