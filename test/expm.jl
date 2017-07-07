@testset "Matrix exponential" begin
    @test expm(@SMatrix [2])::SMatrix ≈ SMatrix{1,1}(expm(2))
    @test expm(@SMatrix [5 2; -2 1])::SMatrix ≈ expm([5 2; -2 1])
    @test expm(@SMatrix [4 2; -2 1])::SMatrix ≈ expm([4 2; -2 1])
    @test expm(@SMatrix [4 2; 2 1])::SMatrix ≈ expm([4 2; 2 1])
    @test expm(@SMatrix [ -3+1im  -1+2im;-4-5im   5-2im])::SMatrix ≈ expm(Complex{Float64}[ -3+1im  -1+2im;-4-5im   5-2im])
    @test expm(@SMatrix [1 2 0; 2 1 0; 0 0 1])::SizedArray{Tuple{3,3}} ≈ expm([1 2 0; 2 1 0; 0 0 1])
end
