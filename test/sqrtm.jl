@testset "Matrix square root" begin
    @test sqrtm(@SMatrix [2])::SMatrix ≈ SMatrix{1,1}(sqrtm(2))
    @test sqrtm(@SMatrix [5 2; -2 1])::SMatrix ≈ sqrtm([5 2; -2 1])
    @test sqrtm(@SMatrix [4 2; -2 1])::SMatrix ≈ sqrtm([4 2; -2 1])
    @test sqrtm(@SMatrix [4 2; 2 1])::SMatrix ≈ sqrtm([4 2; 2 1])
    @test sqrtm(@SMatrix [1 2 0; 2 1 0; 0 0 1])::SizedArray{Tuple{3,3}} ≈ sqrtm([1 2 0; 2 1 0; 0 0 1])
end
