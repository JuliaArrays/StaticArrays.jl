using StaticArrays, Base.Test

@testset "LU decomposition" begin
    # Square case
    m22 = @SMatrix [1 2; 3 4]
    @test @inferred(lu(m22)) isa Tuple{LowerTriangular{Float64,SMatrix{2,2,Float64,4}}, UpperTriangular{Float64,SMatrix{2,2,Float64,4}}, SVector{2,Int}}
    @test lu(m22)[1]::LowerTriangular{<:Any,<:StaticMatrix} ≊ lu(Matrix(m22))[1]
    @test lu(m22)[2]::UpperTriangular{<:Any,<:StaticMatrix} ≊ lu(Matrix(m22))[2]
    @test lu(m22)[3]::StaticVector ≊ lu(Matrix(m22))[3]

    # Rectangular case
    m23 = @SMatrix Float64[3 9 4; 6 6 2]
    @test @inferred(lu(m23)) isa Tuple{SMatrix{2,2,Float64,4}, SMatrix{2,3,Float64,6}, SVector{2,Int}}
    @test lu(m23)[1] ≊ lu(Matrix(m23))[1]
    @test lu(m23)[2] ≊ lu(Matrix(m23))[2]
    @test lu(m23)[3] ≊ lu(Matrix(m23))[3]

    @test lu(m23')[1] ≊ lu(Matrix(m23'))[1]
    @test lu(m23')[2] ≊ lu(Matrix(m23'))[2]
    @test lu(m23')[3] ≊ lu(Matrix(m23'))[3]
end
