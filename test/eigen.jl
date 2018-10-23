using StaticArrays, Test, LinearAlgebra

@testset "Eigenvalue decomposition" begin
    @testset "1×1" begin
        m = @SMatrix [2.0]
        (vals, vecs) = eigen(m)
        @test vals === SVector(2.0)
        @test eigvals(m) === vals
        @test vecs === SMatrix{1,1}(1.0)
        ef = eigen(m)
        @test ef.values === SVector(2.0)
        @test ef.vectors === SMatrix{1,1}(1.0)

        (vals, vecs) = eigen(Symmetric(m))
        @test vals === SVector(2.0)
        @test vecs === SMatrix{1,1}(1.0)
        ef = eigen(Symmetric(m))
        @test ef.values === SVector(2.0)
        @test ef.vectors === SMatrix{1,1}(1.0)
        # handle non-Hermitian case
        m = @SMatrix [2.0+im]
        @test eigvals(m) === SVector(2.0+im)
    end

    @testset "2×2" for i = 1:100
        m_a = randn(2,2)
        @test_throws ErrorException eigvals(SMatrix{2,2}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{2,2}(m_a)

        (vals_a, vecs_a) = eigen(m_a)
        (vals, vecs) = eigen(m)
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m
        ef = eigen(m)
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m
        ef = eigen(Symmetric(m))
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m
        ef = eigen(Symmetric(m, :L))
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m

        (vals, vecs) = eigen(Hermitian(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(Hermitian(m)) ≈ vals
        @test eigvals(Hermitian(m, :L)) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m
        ef = eigen(Hermitian(m))
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m
        ef = eigen(Hermitian(m, :L))
        @test ef.values::SVector ≈ vals_a
        @test (ef.vectors*diagm(Val(0) => vals)*ef.vectors')::SMatrix ≈ m

        m_d = randn(SVector{2}); m = diagm(Val(0) => m_d)
        (vals, vecs) = eigen(Hermitian(m))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eigen(Hermitian(m, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m) ≈ sort(m_d)
        @test eigvals(Hermitian(m)) ≈ sort(m_d)

        # issue #523
        for (i, j) in ((1, 2), (2, 1)), uplo in (:U, :L)
            A = SMatrix{2,2,Float64}((i, 0, 0, j))
            E = eigen(Symmetric(A, uplo))
            @test eigvecs(E) * SDiagonal(eigvals(E)) * eigvecs(E)' ≈ A
        end
    end

    @testset "3×3" for i = 1:100
        m_a = randn(3,3)
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)

        (vals_a, vecs_a) = eigen(m)
        (vals, vecs) = eigen(m)
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test eigvals(Hermitian(m)) ≈ vals
        @test eigvals(Hermitian(m, :L)) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m, :L))
        @test vals::SVector ≈ vals_a

        m_d = randn(SVector{3}); m = diagm(Val(0) => m_d)
        (vals, vecs) = eigen(Hermitian(m))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eigen(Hermitian(m, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m) ≈ sort(m_d)
        @test eigvals(Hermitian(m)) ≈ sort(m_d)
    end

    @testset "3x3 degenerate cases" begin
        # Rank 1
        v = randn(SVector{3,Float64})
        m = v*v'
        vv = sum(abs2, v)
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vecs'*vecs ≈ one(SMatrix{3,3,Float64})
        @test vals ≈ SVector(0.0, 0.0, vv)
        @test eigvals(m) ≈ vals

        # Rank 2
        v2 = randn(SVector{3,Float64})
        v2 -= dot(v,v2)*v/(vv)
        v2v2 = sum(abs2, v2)
        m += v2*v2'
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vecs'*vecs ≈ one(SMatrix{3,3,Float64})
        if vv < v2v2
            @test vals ≈ SVector(0.0, vv, v2v2)
        else
            @test vals ≈ SVector(0.0, v2v2, vv)
        end
        @test eigvals(m) ≈ vals

        # Degeneracy (2 large)
        m = -99*(v*v')/vv + 100*one(SMatrix{3,3,Float64})
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vecs'*vecs ≈ one(SMatrix{3,3,Float64})
        @test vals ≈ SVector(1.0, 100.0, 100.0)
        @test eigvals(m) ≈ vals

        # Degeneracy (2 small)
        m = (v*v')/vv + 1e-2*one(SMatrix{3,3,Float64})
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vecs'*vecs ≈ one(SMatrix{3,3,Float64})
        @test vals ≈ SVector(1e-2, 1e-2, 1.01)
        @test eigvals(m) ≈ vals

        # Block diagonal
        m = @SMatrix [1.0 0.0 0.0;
                      0.0 1.0 1.0;
                      0.0 1.0 1.0]
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs*diagm(Val(0) => vals)*vecs' ≈ m
        @test eigvals(m) ≈ vals

        m = @SMatrix [1.0 0.0 1.0;
                      0.0 1.0 0.0;
                      1.0 0.0 1.0]
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs*diagm(Val(0) => vals)*vecs' ≈ m
        @test eigvals(m) ≈ vals

        m = @SMatrix [1.0 1.0 0.0;
                      1.0 1.0 0.0;
                      0.0 0.0 1.0]
        vals, vecs = eigen(m)::Eigen{<:Any,<:Any,<:SMatrix,<:SVector}

        @test vals ≈ [0.0, 1.0, 2.0]
        @test vecs*diagm(Val(0) => vals)*vecs' ≈ m
        @test eigvals(m) ≈ vals
    end

    @testset "4×4" for i = 1:100
        m_a = randn(4,4)
        m_a = m_a*m_a'
        m = SMatrix{4,4}(m_a)

        (vals_a, vecs_a) = eigen(m_a)
        (vals, vecs) = eigen(m)
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m))
        @test vals::SVector ≈ vals_a
        @test eigvals(m) ≈ vals
        @test eigvals(Hermitian(m)) ≈ vals
        @test eigvals(Hermitian(m, :L)) ≈ vals
        @test (vecs*diagm(Val(0) => vals)*vecs')::SMatrix ≈ m

        (vals, vecs) = eigen(Symmetric(m, :L))
        @test vals::SVector ≈ vals_a
        m_d = randn(SVector{4}); m = diagm(Val(0) => m_d)
        (vals, vecs) = eigen(Hermitian(m))
        @test vals::SVector ≈ sort(m_d)
        (vals, vecs) = eigen(Hermitian(m, :L))
        @test vals::SVector ≈ sort(m_d)
        @test eigvals(m) ≈ sort(m_d)
        @test eigvals(Hermitian(m)) ≈ sort(m_d)

        # not Hermitian
        @test_throws Exception eigen(@SMatrix randn(4,4))
    end

    @testset "complex" begin
        for n=1:5
            a = randn(n,n)+im*randn(n,n)
            a = a+a'
            A = Hermitian(SMatrix{n,n}(a))
            D,V = eigen(A)
            @test V'V ≈ Matrix(I, n, n)
            @test V*diagm(Val(0) => D)*V' ≈ A
            @test V'*A*V ≈ diagm(Val(0) => D)
        end
    end
end
