@testset "Matrix inverse" begin
    @test inv(@SMatrix [2])::SMatrix ≈ @SMatrix [0.5]
    @test inv(@SMatrix [1 2; 2 1])::SMatrix ≈ [-1/3 2/3; 2/3 -1/3]
    @test inv(@SMatrix [1 2 0; 2 1 0; 0 0 1])::SMatrix ≈ [-1/3 2/3 0; 2/3 -1/3 0; 0 0 1]
    m = randn(Float64, 4,4) + eye(4) # well conditioned
    @test inv(SMatrix{4,4}(m))::StaticMatrix ≈ inv(m)

    @test inv(@SMatrix [0x01 0x02; 0x02 0x01])::SMatrix ≈ [-1/3 2/3; 2/3 -1/3]
    @test inv(@SMatrix [0x01 0x02 0x00; 0x02 0x01 0x00; 0x00 0x00 0x01])::SMatrix ≈ [-1/3 2/3 0; 2/3 -1/3 0; 0 0 1]
end
