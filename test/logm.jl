using StaticArrays, Test, LinearAlgebra

@testset "Matrix logarithm" begin
    @test log(@SMatrix [2])::SMatrix ≈ SMatrix{1,1}(log(2))
    @test log(@SMatrix [5 2; -2 1])::SMatrix ≈ log([5 2; -2 1])
    @test log(@SMatrix [4 2; -2 1])::SMatrix ≈ log([4 2; -2 1])
    @test log(@SMatrix [1 2; 3 4])::SMatrix ≈ log([1 2; 3 4])
    @test log(@SMatrix [ -3+1im  -1+2im;-4-5im   5-2im])::SMatrix ≈ log(Complex{Float64}[ -3+1im  -1+2im;-4-5im   5-2im])
    @test log(@SMatrix [1 2 0; 2 1 0; 0 0 1]) ≈ log([1 2 0; 2 1 0; 0 0 1])

    for sz in (3,4), typ in (Float64, Complex{Float64})
        A = rand(typ, sz, sz)
        nA = norm(A, 1)
        for nB in (0.005, 0.1, 0.5, 3.0, 20.0)
            B = A*nB/nA
            SB = SMatrix{sz,sz,typ}(B)
            @test log(B) ≈ log(SB)
        end
    end
end
