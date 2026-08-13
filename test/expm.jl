using StaticArrays, Test, LinearAlgebra

@testset "Matrix exponential" begin
    @test exp(SMatrix{0,0,Int}())::SMatrix === SMatrix{0,0,Float64}()
    @test exp(@SMatrix [2])::SMatrix ≈ SMatrix{1,1}(exp(2))
    @test exp(@SMatrix [5 2; -2 1])::SMatrix ≈ exp([5 2; -2 1])
    @test exp(@SMatrix [4 2; -2 1])::SMatrix ≈ exp([4 2; -2 1])
    @test exp(@SMatrix [4 2; 2 1])::SMatrix ≈ exp([4 2; 2 1])
    @test exp(@SMatrix [ -3+1im  -1+2im;-4-5im   5-2im])::SMatrix ≈ exp(Complex{Float64}[ -3+1im  -1+2im;-4-5im   5-2im])
    # test for case `(a[1,1] - a[2,2])^2 + 4*a[2,1]*a[1,2] == 0`
    @test exp(@SMatrix [2+2im  1-1im; -2-2im  6+2im])::SMatrix ≈ exp(Complex{Float64}[2+2im  1-1im; -2-2im  6+2im])
    # test for zeros matrix
    @test exp(@SMatrix zeros(Complex{Float64}, 2, 2))::SMatrix ≈ Complex{Float64}[1 0; 0 1]
    @test exp(@SMatrix [1 2 0; 2 1 0; 0 0 1]) ≈ exp([1 2 0; 2 1 0; 0 0 1])

    # https://github.com/JuliaArrays/StaticArrays.jl/issues/1295
    @test exp(@SMatrix [-800.0 800.0; 800.0 -800.0])::SMatrix ≈ [0.5 0.5; 0.5 0.5]
    @test exp(@SMatrix [-800.0 800.0; 800.0 -800.0]) ≈ exp([-800.0 800.0; 800.0 -800.0])

    for sz in (3,4), typ in (Float64, Complex{Float64})
        A = rand(typ, sz, sz)
        nA = norm(A, 1)
        for nB in (0.005, 0.1, 0.5, 3.0, 20.0)
            B = A*nB/nA
            SB = SMatrix{sz,sz,typ}(B)
            @test exp(B) ≈ exp(SB)
        end
    end
end
