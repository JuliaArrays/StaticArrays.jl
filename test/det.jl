@testset "Determinant" begin
    @test det(@SMatrix [1]) === 1
    @test logdet(@SMatrix [1]) === 0.0
    @test det(@SMatrix [0 1; 1 0]) === -1
    @test logdet(@SMatrix Complex{Float64}[0 1; 1 0]) == log(det(@SMatrix Complex{Float64}[0 1; 1 0]))

    @test det(@SMatrix [0 1 0; 1 0 0; 0 0 1]) === -1
    m = randn(Float64, 4,4)
    @test det(SMatrix{4,4}(m)) ≈ det(m)
    #triu/tril
    @test det(@SMatrix [1 2; 0 3]) === 3
    @test det(@SMatrix [1 2 3 4; 0 5 6 7; 0 0 8 9; 0 0 0 10]) === 400.0
    @test logdet(@SMatrix [1 2 3 4; 0 5 6 7; 0 0 8 9; 0 0 0 10]) ≈ log(400.0)
    @test @inferred(det(ones(SMatrix{10,10,Complex{Float64}}))) == 0

    # Unsigned specializations , compare to Base
    M = @SMatrix [1 2 3 4; 200 5 6 7; 0 0 8 9; 0 0 0 10]
    for sz in (2,3,4), typ in (UInt8,UInt16,UInt32,UInt64)
        Mtag = SMatrix{sz,sz,typ}(M[1:sz,1:sz])
        @test det(Mtag) == det(Array(Mtag))
    end

    @test_throws DimensionMismatch det(@SMatrix [0; 1])
end
