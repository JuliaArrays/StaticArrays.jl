@testset "Matrix exponential" begin
    @test expm(@SMatrix [2])::SMatrix ≈ SMatrix{1,1}(expm(2))
    @test expm(@SMatrix [5 2; -2 1])::SMatrix ≈ expm([5 2; -2 1])
    @test expm(@SMatrix [4 2; -2 1])::SMatrix ≈ expm([4 2; -2 1])
    @test expm(@SMatrix [4 2; 2 1])::SMatrix ≈ expm([4 2; 2 1])
    @test expm(@SMatrix [ -3+1im  -1+2im;-4-5im   5-2im])::SMatrix ≈ expm(Complex{Float64}[ -3+1im  -1+2im;-4-5im   5-2im])
    @test expm(@SMatrix [1 2 0; 2 1 0; 0 0 1]) ≈ expm([1 2 0; 2 1 0; 0 0 1])

    for sz in (3,4), typ in (Float64, Complex{Float64})
        A = rand(typ, sz, sz)
        nA = norm(A, 1)
        for nB in (0.005, 0.1, 0.5, 3.0, 20.0)
            B = A*nB/nA
            SB = SMatrix{sz,sz,typ}(B)
            @test expm(B) ≈ expm(SB)
        end
    end
end
