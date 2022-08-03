using StaticArrays, Test, LinearAlgebra
using LinearAlgebra: PosDefException

@testset "Cholesky decomposition" begin
    for elty in [Float32, Float64, ComplexF64]
        @testset "1×1" begin
            m = @SMatrix [4.0]
            (c,) = cholesky(m).U
            @test c === 2.0
        end

        @testset "2×2" for i = 1:100
            m_a = randn(elty, 2,2)
            #non hermitian
            @test_throws PosDefException cholesky(SMatrix{2,2}(m_a))
            m_a = m_a*m_a'
            m = SMatrix{2,2}(m_a)
            @test cholesky(Hermitian(m)).U ≈ cholesky(m_a).U
            @test cholesky(Hermitian(m)).L ≈ cholesky(m_a).L
        end

        @testset "3×3" for i = 1:100
            m_a = randn(elty, 3,3)
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
            m_a = randn(elty, 4,4)
            #non hermitian
            @test_throws PosDefException cholesky(SMatrix{4,4}(m_a))
            
            m_a = m_a*m_a'
            m = SMatrix{4,4}(m_a)
            @test cholesky(m).L ≈ cholesky(m_a).L
            @test cholesky(m).U ≈ cholesky(m_a).U
            @test cholesky(Hermitian(m)).L ≈ cholesky(m_a).L
            @test cholesky(Hermitian(m)).U ≈ cholesky(m_a).U
        end

        @testset "large (25x25)" begin
            m_a = randn(elty, 25, 25)
            m_a = m_a*m_a'
            m = SMatrix{25,25}(m_a)
            @test cholesky(m).L ≈ cholesky(m_a).L
        end

        @testset "Inverse" begin
            m_a = randn(elty, 3, 3)
            m_a = m_a*m_a'
            m = SMatrix{3,3}(m_a)
            c = cholesky(m)
            @test (@inferred inv(c)) isa SMatrix{3,3,elty}
            @test inv(c) ≈ SMatrix{3,3}(inv(m_a))
        end

        @testset "Division" begin
            m_a = randn(elty, 3, 3)
            m_a = m_a*m_a'
            m = SMatrix{3,3}(m_a)
            c = cholesky(m)
            c_a = cholesky(m_a)
            
            d_a = randn(elty, 3, 3)
            d = SMatrix{3,3}(d_a)

            @test (@inferred c \ d) isa SMatrix{3,3,elty}
            @test c \ d ≈ c_a \ d_a
            @test (@inferred c \ Symmetric(d)) isa SMatrix{3,3,elty}
            @test c \ Symmetric(d) ≈ c_a \ Symmetric(d_a)

            @test (@inferred d / c) isa SMatrix{3,3,elty}
            @test d / c ≈ d_a / c_a
            @test (@inferred Symmetric(d) / c) isa SMatrix{3,3,elty}
            @test Symmetric(d) / c ≈ Symmetric(d_a) / c_a

            v_a = randn(elty, 3)
            v = SVector{3}(v_a)
            @test (@inferred c \ v) isa SVector{3,elty}
            @test c \ v ≈ c_a \ v_a
        end

        @testset "Check" begin
            for i ∈ [1,3,7,25]
                x = SVector(ntuple(elty, i))
                nonpd = x * x'
                if i > 1
                    @test_throws PosDefException cholesky(nonpd)
                    @test !issuccess(cholesky(nonpd,check=false))
                    @test_throws PosDefException cholesky(Hermitian(nonpd))
                    @test !issuccess(cholesky(Hermitian(nonpd),check=false))
                else
                    @test issuccess(cholesky(Hermitian(nonpd),check=false))
                end
            end
        end
    end

    @testset "static blockmatrix" for i = 1:10
        m_a = randn(3,3)
        m_a = m_a*m_a'
        m = SMatrix{3,3}(m_a)
        @test_broken cholesky(reshape([m, 0m, 0m, m], 2, 2)) ==
            reshape([chol(m), 0m, 0m, chol(m)], 2, 2)
    end
end
