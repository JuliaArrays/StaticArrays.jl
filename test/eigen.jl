@testset "Eigenvalue decomposition" begin
    @testset "1×1" begin
        m = @SMatrix [2.0]
        (vals, vecs) = eig(m)
        @test vals === SVector(2.0)
        @test vecs === SMatrix{1,1}(1.0)

        (vals, vecs) = eig(Symmetric(m))
        @test vals === SVector(2.0)
        @test vecs === SMatrix{1,1}(1.0)
    end

    @testset "2×2" for i = 1:100
        m_a = randn(2,2)
        m_a = m_a*m_a'
        m = SMatrix{2,2}(m_a)

        (vals_a, vecs_a) = eig(m)
        (vals, vecs) = eig(m)
        @test vals::SVector ≈ vals_a
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eig(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m
    end

    @testset "3×3" for i = 1:100
        m_a = randn(3,3)
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)

        (vals_a, vecs_a) = eig(m)
        (vals, vecs) = eig(m)
        @test vals::SVector ≈ vals_a
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eig(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m
    end
end
