@testset "Determinant" begin
    @test det(@SMatrix [1]) === 1
    @test det(@SMatrix [0 1; 1 0]) === -1
    @test det(@SMatrix [0 1 0; 1 0 0; 0 0 1]) === -1
    m = randn(Float64, 4,4)
    @test det(SMatrix{4,4}(m)) ≈ det(m)
    #triu/tril
    @test det(@SMatrix [1 2; 0 3]) === 3
    @test det(@SMatrix [1 2 3 4; 0 5 6 7; 0 0 8 9; 0 0 0 10]) === 400.0

    # Unsigned specializations
    @test det(@SMatrix [0x00 0x01; 0x01 0x00])::Int8 == -1
    @test det(@SMatrix [0x00 0x01 0x00; 0x01 0x00 0x00; 0x00 0x00 0x01])::Int8 == -1

    @test_throws DimensionMismatch det(@SMatrix [0; 1])
end
