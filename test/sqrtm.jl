using StaticArrays, Test

@testset "Matrix square root" begin
    @test sqrt(SMatrix{0,0,Int}())::SMatrix === SMatrix{0,0,Float64}()
    @test sqrt(@SMatrix [2])::SMatrix ≈ SMatrix{1,1}(sqrt(2))
    @test sqrt(@SMatrix [5 2; -2 1])::SMatrix ≈ sqrt([5 2; -2 1])
    @test sqrt(@SMatrix [4 2; -2 1])::SMatrix ≈ sqrt([4 2; -2 1])
    @test sqrt(@SMatrix [4 2; 2 1])::SMatrix ≈ sqrt([4 2; 2 1])
    @test sqrt(@SMatrix [0 0; 0 0])::SMatrix ≈ sqrt([0 0; 0 0])
    @test sqrt(@SMatrix [1 2 0; 2 1 0; 0 0 1])::SizedArray{Tuple{3,3}} ≈ sqrt([1 2 0; 2 1 0; 0 0 1])
end
