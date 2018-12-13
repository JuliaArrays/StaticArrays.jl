using StaticArrays, Test, LinearAlgebra
using LinearAlgebra: PosDefException

@testset "Cholesky decomposition" begin
    @testset "1×1" begin
        m = @SMatrix [4.0]
        (c,) = cholesky(m).U
        @test c === 2.0
    end

    @testset "2×2" for i = 1:100
        m_a = randn(2,2)
        #non hermitian
        @test_throws PosDefException cholesky(SMatrix{2,2}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{2,2}(m_a)
        @test cholesky(Hermitian(m)).U ≈ cholesky(m_a).U
        @test cholesky(Hermitian(m)).L ≈ cholesky(m_a).L
    end

    @testset "3×3" for i = 1:100
        m_a = randn(3,3)
        #non hermitian
        @test_throws PosDefException cholesky(SMatrix{3,3}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)
        @test cholesky(m).U ≈ cholesky(m_a).U
        @test cholesky(m).L ≈ cholesky(m_a).L
        @test cholesky(Hermitian(m)).U ≈ cholesky(m_a).U
        @test cholesky(Hermitian(m)).L ≈ cholesky(m_a).L
    end
    @testset "4×4" for i = 1:100
        m_a = randn(4,4)
        #non hermitian
        @test_throws PosDefException cholesky(SMatrix{4,4}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{4,4}(m_a)
        @test cholesky(m).L ≈ cholesky(m_a).L
        @test cholesky(m).U ≈ cholesky(m_a).U
        @test cholesky(Hermitian(m)).L ≈ cholesky(m_a).L
        @test cholesky(Hermitian(m)).U ≈ cholesky(m_a).U
    end
    @testset "static blockmatrix" for i = 1:10
        m_a = randn(3,3)
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)
        @test_broken cholesky(reshape([m, 0m, 0m, m], 2, 2)) ==
            reshape([chol(m), 0m, 0m, chol(m)], 2, 2)
    end
end
