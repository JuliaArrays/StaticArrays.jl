using StaticArrays, Base.Test

@testset "LU decomposition" begin
    # Square case
    m22 = @SMatrix [1 2; 3 4]
    @inferred(lu(m22))
    @test lu(m22)[1]::LowerTriangular{<:Any,<:StaticMatrix} ≊ lu(Matrix(m22))[1]
    @test lu(m22)[2]::UpperTriangular{<:Any,<:StaticMatrix} ≊ lu(Matrix(m22))[2]
    @test lu(m22)[3]::StaticVector ≊ lu(Matrix(m22))[3]

    # Rectangular case
    m23 = @SMatrix Float64[3 9 4; 6 6 2]
    @inferred lu(m23)
    @test lu(m23)[1] ≊ lu(Matrix(m23))[1]
    @test lu(m23)[2] ≊ lu(Matrix(m23))[2]
    @test lu(m23)[3] ≊ lu(Matrix(m23))[3]

    @test lu(m23')[1] ≊ lu(Matrix(m23'))[1]
    @test lu(m23')[2] ≊ lu(Matrix(m23'))[2]
    @test lu(m23')[3] ≊ lu(Matrix(m23'))[3]
end
