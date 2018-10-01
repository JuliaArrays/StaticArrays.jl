using StaticArrays, Test, LinearAlgebra

@testset "Matrix square root" begin
    @test sqrt(@SMatrix [2])::SMatrix ≈ SMatrix{1,1}(sqrt(2))
    @test sqrt(@SMatrix [5 2; -2 1])::SMatrix ≈ sqrt([5 2; -2 1])
    @test sqrt(@SMatrix [4 2; -2 1])::SMatrix ≈ sqrt([4 2; -2 1])
    @test sqrt(@SMatrix [4 2; 2 1])::SMatrix ≈ sqrt([4 2; 2 1])
    @test sqrt(@SMatrix [0 0; 0 0])::SMatrix ≈ sqrt([0 0; 0 0])
    @test sqrt(@SMatrix [0 4; 4 0])::SMatrix ≈ sqrt([0 4; 4 0]) # tests for square root of a negative value
    @test sqrt(@SMatrix [1 2 0; 2 1 0; 0 0 1])::SizedArray{Tuple{3,3}} ≈ sqrt([1 2 0; 2 1 0; 0 0 1])
    @test sqrt(@SMatrix [1 2 0; 2 1 0; 5 0 1])::SMatrix ≈ sqrt([1 2 0; 2 1 0; 5 0 1])

    for sz in (2,3,4), typ in (Float64, Complex{Float64})
        A = rand(typ, sz, sz)
        nA = norm(A, 1)
        for nB in (0.005, 0.1, 0.5, 3.0, 20.0)
            B = A*nB/nA
            SB = SMatrix{sz,sz,typ}(B)
            @test sqrt(B) ≈ sqrt(SB)
        end
    end
end
