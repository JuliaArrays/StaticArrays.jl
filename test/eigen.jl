@testset "Eigenvalue decomposition" begin
    @testset "1×1" begin
        m = @SMatrix [2.0]
        (vals, vecs) = eig(m)
        @test vals === SVector(2.0)
        @test eigvals(m) === vals
        @test vecs === SMatrix{1,1}(1.0)
        ef = eigfact(m)
        @test ef[:values] === SVector(2.0)
        @test ef[:vectors] === SMatrix{1,1}(1.0)

        (vals, vecs) = eig(Symmetric(m))
        @test vals === SVector(2.0)
        @test vecs === SMatrix{1,1}(1.0)
        ef = eigfact(Symmetric(m))
        @test ef[:values] === SVector(2.0)
        @test ef[:vectors] === SMatrix{1,1}(1.0)
    end

    @testset "2×2" for i = 1:100
        m_a = randn(2,2)
        @test_throws ErrorException eigvals(SMatrix{2,2}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{2,2}(m_a)

        (vals_a, vecs_a) = eig(m_a)
        (vals, vecs) = eig(m)
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m
        ef = eigfact(m)
        @test ef[:values]::SVector ≈ vals_a
        @test (ef[:vectors]*diagm(vals)*ef[:vectors]')::SMatrix ≈ m

        (vals, vecs) = eig(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m
        ef = eigfact(Symmetric(m))
        @test ef[:values]::SVector ≈ vals_a
        @test (ef[:vectors]*diagm(vals)*ef[:vectors]')::SMatrix ≈ m

        (vals, vecs) = eig(Hermitian(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(Hermitian(m)) ≈ vals
        @test eigvals(Hermitian(m, :L)) ≈ vals
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m
        ef = eigfact(Hermitian(m))
        @test ef[:values]::SVector ≈ vals_a
        @test (ef[:vectors]*diagm(vals)*ef[:vectors]')::SMatrix ≈ m

        m_d = randn(SVector{2}); m = diagm(m_d)
        (vals, vecs) = eig(Hermitian(m))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eig(Hermitian(m, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m) ≈ sort(m_d)
        @test eigvals(Hermitian(m)) ≈ sort(m_d)
    end

    @testset "3×3" for i = 1:100
        m_a = randn(3,3)
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)

        (vals_a, vecs_a) = eig(m)
        (vals, vecs) = eig(m)
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eig(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test eigvals(Hermitian(m)) ≈ vals
        @test eigvals(Hermitian(m, :L)) ≈ vals
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eig(Symmetric(m, :L))
        @test vals::SVector ≈ vals_a

        m_d = randn(SVector{3}); m = diagm(m_d)
        (vals, vecs) = eig(Hermitian(m))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eig(Hermitian(m, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m) ≈ sort(m_d)
        @test eigvals(Hermitian(m)) ≈ sort(m_d)
    end

    @testset "3x3 degenerate cases" begin
        # Rank 1
        v = randn(SVector{3,Float64})
        m = v*v'
        vv = sum(abs2, v)
        vals, vecs = eig(m)::Tuple{SVector,SMatrix}

        @test vecs'*vecs ≈ eye(SMatrix{3,3,Float64})
        @test vals ≈ SVector(0.0, 0.0, vv)
        @test eigvals(m) ≈ vals

        # Rank 2
        v2 = randn(SVector{3,Float64})
        v2 -= dot(v,v2)*v/(vv)
        v2v2 = sum(abs2, v2)
        m += v2*v2'
        vals, vecs = eig(m)::Tuple{SVector,SMatrix}

        @test vecs'*vecs ≈ eye(SMatrix{3,3,Float64})
        if vv < v2v2
            @test vals ≈ SVector(0.0, vv, v2v2)
        else
            @test vals ≈ SVector(0.0, v2v2, vv)
        end
        @test eigvals(m) ≈ vals

        # Degeneracy (2 large)
        m = -99*(v*v')/vv + 100*eye(SMatrix{3,3,Float64})
        vals, vecs = eig(m)::Tuple{SVector,SMatrix}

        @test vecs'*vecs ≈ eye(SMatrix{3,3,Float64})
        @test vals ≈ SVector(1.0, 100.0, 100.0)
        @test eigvals(m) ≈ vals

        # Degeneracy (2 small)
        m = (v*v')/vv + 1e-2*eye(SMatrix{3,3,Float64})
        vals, vecs = eig(m)::Tuple{SVector,SMatrix}

        @test vecs'*vecs ≈ eye(SMatrix{3,3,Float64})
        @test vals ≈ SVector(1e-2, 1e-2, 1.01)
        @test eigvals(m) ≈ vals

        # Block diagonal
        m = @SMatrix [1.0 0.0 0.0;
                      0.0 1.0 1.0;
                      0.0 1.0 1.0]
        vals, vecs = eig(m)::Tuple{SVector,SMatrix}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs*diagm(vals)*vecs' ≈ m
        @test eigvals(m) ≈ vals

        m = @SMatrix [1.0 0.0 1.0;
                      0.0 1.0 0.0;
                      1.0 0.0 1.0]
        vals, vecs = eig(m)::Tuple{SVector,SMatrix}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs*diagm(vals)*vecs' ≈ m
        @test eigvals(m) ≈ vals

        m = @SMatrix [1.0 1.0 0.0;
                      1.0 1.0 0.0;
                      0.0 0.0 1.0]
        vals, vecs = eig(m)::Tuple{SVector,SMatrix}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs*diagm(vals)*vecs' ≈ m
        @test eigvals(m) ≈ vals
    end

    @testset "4×4" for i = 1:100
        m_a = randn(4,4)
        m_a = m_a*m_a'
        m = SMatrix{4,4}(m_a)

        (vals_a, vecs_a) = eig(m_a)
        (vals, vecs) = eig(m)
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eig(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test eigvals(Hermitian(m)) ≈ vals
        @test eigvals(Hermitian(m, :L)) ≈ vals
        @test (vecs*diagm(vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eig(Symmetric(m, :L))
        @test vals::SVector ≈ vals_a
        m_d = randn(SVector{4}); m = diagm(m_d)
        (vals, vecs) = eig(Hermitian(m))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eig(Hermitian(m, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m) ≈ sort(m_d)
        @test eigvals(Hermitian(m)) ≈ sort(m_d)

        # not Hermitian
        @test_throws Exception eig(@SMatrix randn(4,4))
    end
end
