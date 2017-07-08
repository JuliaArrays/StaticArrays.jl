@testset "Cholesky decomposition" begin
    @testset "1×1" begin
        m = @SMatrix [4.0]
        (c,) = chol(m)
        @test c === 2.0
    end

    @testset "2×2" for i = 1:100
        m_a = randn(2,2)
        #non hermitian
        @test_throws ArgumentError chol(SMatrix{2,2}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{2,2}(m_a)
        @test chol(Hermitian(m)) ≈ chol(m_a)
    end

    @testset "3×3" for i = 1:100
        m_a = randn(3,3)
        #non hermitian
        @test_throws ArgumentError chol(SMatrix{3,3}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)
        @test chol(m) ≈ chol(m_a)
        @test chol(Hermitian(m)) ≈ chol(m_a)
    end
    @testset "4×4" for i = 1:100
        m_a = randn(4,4)
        #non hermitian
        @test_throws ArgumentError chol(SMatrix{4,4}(m_a))
        m_a = m_a*m_a'
        m = SMatrix{4,4}(m_a)
        @test chol(m) ≈ chol(m_a)
        @test chol(Hermitian(m)) ≈ chol(m_a)
    end
end

@testset "Cholesky decomposition" begin
    @testset "eltype <: Real" begin
        for n = (1, 2, 3, 4)
            A = randn(n,n) |> t -> t't
            @test chol(SMatrix{n,n}(A)) ≈ chol(A)
            CU = cholfact(Symmetric(A))
            SCU = cholfact(Symmetric(SMatrix{n,n}(A)))
            @test SCU.uplo == CU.uplo
            @test SCU.factors ≈ CU.factors
            CL = cholfact(Symmetric(A, :L))
            SCL = cholfact(Symmetric(SMatrix{n,n}(A), :L))
            @test SCL.uplo == CL.uplo
            @test SCL.factors ≈ CL.factors
        end
    end

    @testset "eltype <: Complex" begin
        for n = (1, 2, 3, 4)
            A = complex.(randn(n,n), randn(n,n)) |> t -> t't
            @test chol(SMatrix{n,n}(A)) ≈ chol(A)
            CU = cholfact(Hermitian(A))
            SCU = cholfact(Hermitian(SMatrix{n,n}(A)))
            @test SCU.uplo == CU.uplo
            @test SCU.factors ≈ CU.factors
            CL = cholfact(Hermitian(A, :L))
            SCL = cholfact(Hermitian(SMatrix{n,n}(A), :L))
            @test SCL.uplo == CL.uplo
            @test SCL.factors ≈ CL.factors
        end
    end

    @testset "Throw if non-Hermitian" begin
        R = randn(4,4)
        C = complex.(R, R)
        for A in (R, C)
            @test_throws ArgumentError cholfact(A)
            @test_throws ArgumentError chol(A)
        end
    end
end

@testset "Solve linear system" begin
    A = @SMatrix [4. 12. -16.; 12. 37. -43.; -16. -43. 98.]
    B = @SVector [0., 6., 39.]
    C = cholfact(A)
    @test @inferred(C \ B) === ones(SVector{3})
end
