using StaticArrays, Test, LinearAlgebra

@testset "Matrix logarithm" begin
    @test log(SMatrix{0,0,Int}())::SMatrix == SMatrix{0,0,Bool}()
    @test log(@SMatrix [2])::SMatrix ≈ SMatrix{1,1}(log(2))
    @test log(@SMatrix [5 2; -2 1])::SMatrix ≈ log([5 2; -2 1])
    @test log(@SMatrix [4 2; -2 1])::SMatrix ≈ log([4 2; -2 1])
    @test log(@SMatrix [4 2; 2 1])::SMatrix ≈ log([4 2; 2 1])
    # log(::Matrix) is inaccurate on at least Julia 1.1...1.2. We need
    # to record the result explicitly.
    # @test log(@SMatrix [-3+1im -1+2im; -4-5im 5-2im])::SMatrix ≈ log(Complex{Float64}[-3+1im -1+2im; -4-5im 5-2im])
    @test log(@SMatrix [-3+1im -1+2im; -4-5im 5-2im])::SMatrix ≈
        [1.6949262962402925 + 2.56908641247419im 0.4359384105167865 + 0.47910454835485583im
         -1.7687979183427713 + 0.558514409317816im 1.7199705725159193 + 0.09415380973694587im]
    # test for identity matrix
    @test log(SMatrix{2,2}(I))::SMatrix ≈ zeros(2,2)
    @test log(@SMatrix Complex{Float64}[1 2 0; 2 1 0; 0 0 1]) ≈ log([1 2 0; 2 1 0; 0 0 1])

    for sz in 0:4, typ in (Float64, Complex{Float64})
        A = rand(SMatrix{sz,sz,typ})
        expA = exp(A)
        logexpA = log(expA)
        @test logexpA ≈ A
    end
end
