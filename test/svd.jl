using StaticArrays, Base.Test

@testset "SVD factorization" begin
    m3 = @SMatrix Float64[3 9 4; 6 6 2; 3 7 9]
    m3c = Complex128.(m3)
    m23 = @SMatrix Float64[3 9 4; 6 6 2]

    @testset "svd" begin
        @testinf svdvals(@SMatrix [2 0; 0 0])::StaticVector ≊ [2, 0]
        @testinf svdvals((@SMatrix [2 -2; 1 1]) / sqrt(2)) ≊ [2, 1]

        @testinf svdvals(m3) ≊ svdvals(Matrix(m3))
        @testinf svdvals(m3c) isa SVector{3,Float64}

        @testinf svd(m3)[1]::StaticMatrix ≊ svd(Matrix(m3))[1]
        @testinf svd(m3)[2]::StaticVector ≊ svd(Matrix(m3))[2]
        @testinf svd(m3)[3]::StaticMatrix ≊ svd(Matrix(m3))[3]
    end

    @testset "svdfact" begin
        @test_throws ErrorException svdfact(@SMatrix [1 0; 0 1])[:U]

        @testinf svdfact(@SMatrix [2 0; 0 0]).U === one(SMatrix{2,2})
        @testinf svdfact(@SMatrix [2 0; 0 0]).S === SVector(2.0, 0.0)
        @testinf svdfact(@SMatrix [2 0; 0 0]).Vt === one(SMatrix{2,2})

        @testinf svdfact((@SMatrix [2 -2; 1 1]) / sqrt(2)).U ≊ [-1 0; 0 1]
        @testinf svdfact((@SMatrix [2 -2; 1 1]) / sqrt(2)).S ≊ [2, 1]
        @testinf svdfact((@SMatrix [2 -2; 1 1]) / sqrt(2)).Vt ≊ [-1 1; 1 1]/sqrt(2)

        @testinf svdfact(m23).U  ≊ svdfact(Matrix(m23))[:U]
        @testinf svdfact(m23).S  ≊ svdfact(Matrix(m23))[:S]
        @testinf svdfact(m23).Vt ≊ svdfact(Matrix(m23))[:Vt]

        @testinf svdfact(m23').U  ≊ svdfact(Matrix(m23'))[:U]
        @testinf svdfact(m23').S  ≊ svdfact(Matrix(m23'))[:S]
        @testinf svdfact(m23').Vt ≊ svdfact(Matrix(m23'))[:Vt]

        @testinf svdfact(m3c).U  ≊ svdfact(Matrix(m3c))[:U]
        @testinf svdfact(m3c).S  ≊ svdfact(Matrix(m3c))[:S]
        @testinf svdfact(m3c).Vt ≊ svdfact(Matrix(m3c))[:Vt]

        @testinf svdfact(m3c).U  isa SMatrix{3,3,Complex128}
        @testinf svdfact(m3c).S  isa SVector{3,Float64}
        @testinf svdfact(m3c).Vt isa SMatrix{3,3,Complex128}
    end
end
