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

    @testset "3x3 degenerate cases" begin
        # Rank 1
        v = randn(SVector{3,Float64})
        m = v*v'
        eigvals, eigvecs = eig(m)::Tuple{SVector,SMatrix}

        @test isapprox(eigvecs'*eigvecs, eye(SMatrix{3,3,Float64}); atol = 1e-4) # This algorithm isn't super accurate
        @test eigvals ≈ SVector(0.0, 0.0, sumabs2(v))

        # Rank 2
        v2 = randn(SVector{3,Float64})
        v2 -= dot(v,v2)*v/sumabs2(v)
        m += v2*v2'
        eigvals, eigvecs = eig(m)::Tuple{SVector,SMatrix}

        @test isapprox(eigvecs'*eigvecs, eye(SMatrix{3,3,Float64}); atol = 1e-4)
        if sumabs2(v) < sumabs2(v2)
            @test eigvals ≈ SVector(0.0, sumabs2(v), sumabs2(v2))
        else
            @test eigvals ≈ SVector(0.0, sumabs2(v2), sumabs2(v))
        end

        # Degeneracy (2 large)
        m = -99*(v*v')/sumabs2(v) + 100*eye(SMatrix{3,3,Float64})
        eigvals, eigvecs = eig(m)::Tuple{SVector,SMatrix}

        @test isapprox(eigvecs'*eigvecs, eye(SMatrix{3,3,Float64}); atol = 1e-4)
        @test eigvals ≈ SVector(1.0, 100.0, 100.0)

        # Degeneracy (2 small)
        m = (v*v')/sumabs2(v) + 1e-2*eye(SMatrix{3,3,Float64})
        eigvals, eigvecs = eig(m)::Tuple{SVector,SMatrix}

        @test isapprox(eigvecs'*eigvecs, eye(SMatrix{3,3,Float64}); atol = 1e-4)
        @test eigvals ≈ SVector(1e-2, 1e-2, 1.01)

        # Block diagonal
        m = @SMatrix [1.0 0.0 0.0;
                      0.0 1.0 1.0;
                      0.0 1.0 1.0]
        eigvals, eigvecs = eig(m)::Tuple{SVector,SMatrix}

        @test eigvals ≈ [0.0, 1.0, 2.0]
        @test eigvecs*diagm(eigvals)*eigvecs' ≈ m

        m = @SMatrix [1.0 0.0 1.0;
                      0.0 1.0 0.0;
                      1.0 0.0 1.0]
        eigvals, eigvecs = eig(m)::Tuple{SVector,SMatrix}

        @test eigvals ≈ [0.0, 1.0, 2.0]
        @test eigvecs*diagm(eigvals)*eigvecs' ≈ m

        m = @SMatrix [1.0 1.0 0.0;
                      1.0 1.0 0.0;
                      0.0 0.0 1.0]
        eigvals, eigvecs = eig(m)::Tuple{SVector,SMatrix}

        @test eigvals ≈ [0.0, 1.0, 2.0]
        @test eigvecs*diagm(eigvals)*eigvecs' ≈ m
    end
end
