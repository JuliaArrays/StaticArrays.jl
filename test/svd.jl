using StaticArrays, Test, LinearAlgebra

@testset "SVD factorization" begin
    m3 = @SMatrix Float64[3 9 4; 6 6 2; 3 7 9]
    m3_f32 = @SMatrix Float32[3 9 4; 6 6 2; 3 7 9]
    m3c = ComplexF64.(m3)
    m23 = @SMatrix Float64[3 9 4; 6 6 2]
    m23_f32 = @SMatrix Float32[3 9 4; 6 6 2]
    m_sing = @SMatrix [2.0 3.0 5.0; 4.0 6.0 10.0; 1.0 1.0 1.0]
    m_sing2 = @SMatrix [1 1; 1 0; 0 1]
    v = @SVector [1, 2, 3]
    v2 = @SVector [1, 2]
    mc_sing = @SMatrix [1.0+0.1im 0 0; 2.0+0.2im 0 0; 3.0 0.1im 0]
    mc_sing2 = @SMatrix [1.0+0.1im 0; 2.0+0.2im 1; 3.0 0.1im]
    vc = @SVector [1.0f0+8.0f0im, 0.2f0im, 2.5f0]
    vc2 = @SVector [1.0f0+8.0f0im, 0.2f0im]

    @testset "svd" begin
        @testinf svdvals(@SMatrix [2 0; 0 0])::StaticVector ≊ [2, 0]
        @testinf svdvals((@SMatrix [2 -2; 1 1]) / sqrt(2)) ≊ [2, 1]

        @testinf svdvals(m3) ≊ svdvals(Matrix(m3))
        @testinf svdvals(m3_f32) ≊ svdvals(Matrix(m3_f32))
        @testinf svdvals(m3c) isa SVector{3,Float64}

        @testinf svd(m3).U::StaticMatrix ≊ svd(Matrix(m3)).U
        @testinf svd(m3).S::StaticVector ≊ svd(Matrix(m3)).S
        @testinf svd(m3).V::StaticMatrix ≊ svd(Matrix(m3)).V
        @testinf svd(m3).Vt::StaticMatrix ≊ svd(Matrix(m3)).Vt

        @test svd(m3_f32).U::StaticMatrix ≈ svd(Matrix(m3_f32)).U atol = 5e-7
        @test svd(m3_f32).S::StaticVector ≈ svd(Matrix(m3_f32)).S atol = 5e-7
        @test svd(m3_f32).V::StaticMatrix ≈ svd(Matrix(m3_f32)).V atol = 5e-7
        @test svd(m3_f32).Vt::StaticMatrix ≈ svd(Matrix(m3_f32)).Vt atol = 5e-7

        @testinf svd(@SMatrix [2 0; 0 0]).U ≊ one(SMatrix{2,2})
        @testinf svd(@SMatrix [2 0; 0 0]).S ≊ SVector(2.0, 0.0)
        @testinf svd(@SMatrix [2 0; 0 0]).Vt ≊ one(SMatrix{2,2})

        @testinf svd((@SMatrix [2 -2; 1 1]) / sqrt(2)).U ≊ [-1 0; 0 1]
        @testinf svd((@SMatrix [2 -2; 1 1]) / sqrt(2)).S ≊ [2, 1]
        @testinf svd((@SMatrix [2 -2; 1 1]) / sqrt(2)).Vt ≊ [-1 1; 1 1]/sqrt(2)

        @testinf svd(m23).U  ≊ svd(Matrix(m23)).U
        @testinf svd(m23).S  ≊ svd(Matrix(m23)).S
        @testinf svd(m23).Vt ≊ svd(Matrix(m23)).Vt

        @testinf svd(m23').U  ≊ svd(Matrix(m23')).U
        @testinf svd(m23').S  ≊ svd(Matrix(m23')).S
        @testinf svd(m23').Vt ≊ svd(Matrix(m23')).Vt

        @test svd(m23_f32).U::StaticMatrix ≈ svd(Matrix(m23_f32)).U atol = 5e-7
        @test svd(m23_f32).S::StaticVector ≈ svd(Matrix(m23_f32)).S atol = 5e-7
        @test svd(m23_f32).V::StaticMatrix ≈ svd(Matrix(m23_f32)).V atol = 5e-7
        @test svd(m23_f32).Vt::StaticMatrix ≈ svd(Matrix(m23_f32)).Vt atol = 5e-7

        @test svd(m23_f32').U::StaticMatrix ≈ svd(Matrix(m23_f32')).U atol = 5e-7
        @test svd(m23_f32').S::StaticVector ≈ svd(Matrix(m23_f32')).S atol = 5e-7
        @test svd(m23_f32').V::StaticMatrix ≈ svd(Matrix(m23_f32')).V atol = 5e-7
        @test svd(m23_f32').Vt::StaticMatrix ≈ svd(Matrix(m23_f32')).Vt atol = 5e-7

        @testinf svd(m23, full=true).U::StaticMatrix  ≊ svd(Matrix(m23), full=true).U
        @testinf svd(m23, full=true).S::StaticVector  ≊ svd(Matrix(m23), full=true).S
        @testinf svd(m23, full=true).Vt::StaticMatrix ≊ svd(Matrix(m23), full=true).Vt
        @testinf svd(m23', full=true).U::StaticMatrix  ≊ svd(Matrix(m23'), full=true).U
        @testinf svd(m23', full=true).S::StaticVector  ≊ svd(Matrix(m23'), full=true).S
        @testinf svd(m23', full=true).Vt::StaticMatrix ≊ svd(Matrix(m23'), full=true).Vt

        @testinf svd(transpose(m23)).U  ≊ svd(Matrix(transpose(m23))).U
        @testinf svd(transpose(m23)).S  ≊ svd(Matrix(transpose(m23))).S
        @testinf svd(transpose(m23)).Vt ≊ svd(Matrix(transpose(m23))).Vt

        @testinf svd(m3c).U  ≊ svd(Matrix(m3c)).U
        @testinf svd(m3c).S  ≊ svd(Matrix(m3c)).S
        @testinf svd(m3c).Vt ≊ svd(Matrix(m3c)).Vt

        @testinf svd(m3c).U  isa SMatrix{3,3,ComplexF64}
        @testinf svd(m3c).S  isa SVector{3,Float64}
        @testinf svd(m3c).Vt isa SMatrix{3,3,ComplexF64}

        @testinf svd(m3) \ v ≈ svd(Matrix(m3)) \ Vector(v)
        @testinf svd(m_sing) \ v ≈ svd(Matrix(m_sing)) \ Vector(v)
        @testinf svd(m_sing2) \ v ≈ svd(Matrix(m_sing2)) \ Vector(v)
        @testinf svd(m_sing2') \ v2 ≈ svd(Matrix(m_sing2')) \ Vector(v2)
        @testinf svd(m3) \ m23' ≈ svd(Matrix(m3)) \ Matrix(m23')
        @testinf svd(m_sing) \ m23' ≈ svd(Matrix(m_sing)) \ Matrix(m23')
        @testinf svd(m_sing2) \ m23' ≈ svd(Matrix(m_sing2)) \ Matrix(m23')
        @testinf svd(m_sing2; full=Val(true)) \ v ≈ svd(Matrix(m_sing2); full=true) \ Vector(v)
        @testinf svd(m_sing2'; full=Val(true)) \ v2 ≈ svd(Matrix(m_sing2'); full=true) \ Vector(v2)
        @testinf svd(m_sing2; full=Val(true)) \ m23' ≈ svd(Matrix(m_sing2); full=true) \ Matrix(m23')
        @testinf svd(m_sing2'; full=Val(true)) \ m23 ≈ svd(Matrix(m_sing2'); full=true) \ Matrix(m23)

        # Test that svd of rectangular matrix is inferred.
        # Note the placement of @inferred brackets is important.
        #
        # This only seems to work on v"1.5" due to unknown compiler improvements; seems
        # to have stopped working again on v"1.6" and later?
        svd_full_false(A) = svd(A, full=false)
        if VERSION < v"1.10-"
            @test svd_full_false(m_sing2).S ≈ svd(Matrix(m_sing2)).S
        else
            @test @inferred(svd_full_false(m_sing2)).S ≈ svd(Matrix(m_sing2)).S
        end

        @testinf svd(mc_sing) \ v ≈ svd(Matrix(mc_sing)) \ Vector(v)
        @testinf svd(mc_sing) \ vc ≈ svd(Matrix(mc_sing)) \ Vector(vc)
        @testinf svd(mc_sing) \ m23' ≈ svd(Matrix(mc_sing)) \ Matrix(m23')
        @testinf svd(mc_sing2) \ v ≈ svd(Matrix(mc_sing2)) \ Vector(v)
        @testinf svd(mc_sing2) \ vc ≈ svd(Matrix(mc_sing2)) \ Vector(vc)
        @testinf svd(mc_sing2') \ vc2 ≈ svd(Matrix(mc_sing2')) \ Vector(vc2)
        @testinf svd(mc_sing2) \ m23' ≈ svd(Matrix(mc_sing2)) \ Matrix(m23')
        @testinf svd(mc_sing2') \ m23 ≈ svd(Matrix(mc_sing2')) \ Matrix(m23)
        @testinf svd(mc_sing2; full=Val(true)) \ v ≈ svd(Matrix(mc_sing2); full=true) \ Vector(v)
        @testinf svd(mc_sing2; full=Val(true)) \ vc ≈ svd(Matrix(mc_sing2); full=true) \ Vector(vc)
        @testinf svd(mc_sing2'; full=Val(true)) \ vc2 ≈ svd(Matrix(mc_sing2'); full=true) \ Vector(vc2)
        @testinf svd(mc_sing2; full=Val(true)) \ m23' ≈ svd(Matrix(mc_sing2); full=true) \ Matrix(m23')
        @testinf svd(mc_sing2'; full=Val(true)) \ m23 ≈ svd(Matrix(mc_sing2'); full=true) \ Matrix(m23)
    end
end
