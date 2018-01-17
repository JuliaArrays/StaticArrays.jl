using StaticArrays, Base.Test

@testset "Linear algebra" begin

    @testset "SVector as a (mathematical) vector space" begin
        c = 2
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]

        @test @inferred(v1 + c) === @SVector [4,6,8,10]
        @test @inferred(v1 - c) === @SVector [0,2,4,6]
        @test @inferred(v1 * c) === @SVector [4,8,12,16]
        @test @inferred(v1 / c) === @SVector [1.0,2.0,3.0,4.0]
        @test @inferred(c \ v1)::SVector ≈ @SVector [1.0,2.0,3.0,4.0]

        @test @inferred(v1 + v2) === @SVector [6, 7, 8, 9]
        @test @inferred(v1 - v2) === @SVector [-2, 1, 4, 7]

        v3 = [2,4,6,8]
        v4 = [4,3,2,1]

        @test @inferred(v1 + v4) === @SVector [6, 7, 8, 9]
        @test @inferred(v3 + v2) === @SVector [6, 7, 8, 9]
        @test @inferred(v1 - v4) === @SVector [-2, 1, 4, 7]
        @test @inferred(v3 - v2) === @SVector [-2, 1, 4, 7]
    end

    @testset "Interaction with `UniformScaling`" begin
        @test @inferred(@SMatrix([0 1; 2 3]) + I) === @SMatrix [1 1; 2 4]
        @test @inferred(I + @SMatrix([0 1; 2 3])) === @SMatrix [1 1; 2 4]
        @test @inferred(@SMatrix([0 1; 2 3]) - I) === @SMatrix [-1 1; 2 2]
        @test @inferred(I - @SMatrix([0 1; 2 3])) === @SMatrix [1 -1; -2 -2]
        @test_throws DimensionMismatch I + @SMatrix [0 1 4; 2 3 5]

        @test @inferred(@SMatrix([0 1; 2 3]) * I) === @SMatrix [0 1; 2 3]
        @test @inferred(I * @SMatrix([0 1; 2 3])) === @SMatrix [0 1; 2 3]
        @test @inferred(@SMatrix([0 1; 2 3]) / I) === @SMatrix [0.0 1.0; 2.0 3.0]
        @test @inferred(I \ @SMatrix([0 1; 2 3])) === @SMatrix [0.0 1.0; 2.0 3.0]
    end

    @testset "diagm()" begin
        @test @inferred(diagm(SVector(1,2))) === @SMatrix [1 0; 0 2]
        @test @inferred(diagm(SVector(1,2,3), Val{2}))::SMatrix == diagm([1,2,3], 2)
        @test @inferred(diagm(SVector(1,2,3), Val{-2}))::SMatrix == diagm([1,2,3], -2)
    end

    @testset "diag()" begin
        @test @inferred(diag(@SMatrix([0 1; 2 3]))) === SVector(0, 3)
        @test @inferred(diag(@SMatrix([0 1 2; 3 4 5]), Val{1})) === SVector(1, 5)
        @test @inferred(diag(@SMatrix([0 1; 2 3; 4 5]), Val{-1})) === SVector(2, 5)
    end

    @testset "one() and zero()" begin
        @test @inferred(one(SMatrix{2,2,Int})) === @SMatrix [1 0; 0 1]
        @test @inferred(one(SMatrix{2,2})) === @SMatrix [1.0 0.0; 0.0 1.0]
        @test @inferred(one(SMatrix{2})) === @SMatrix [1.0 0.0; 0.0 1.0]
        @test @inferred(one(one(SMatrix{2,2,Int}))) === @SMatrix [1 0; 0 1]
        @test @inferred(zero(SMatrix{2,2,Int})) === @SMatrix [0 0; 0 0]
        @test @inferred(zero(one(SMatrix{2,2,Int}))) === @SMatrix [0 0; 0 0]

        @test @inferred(one(MMatrix{2,2,Int}))::MMatrix == @MMatrix [1 0; 0 1]
        @test @inferred(one(MMatrix{2,2}))::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]
        @test @inferred(one(MMatrix{2}))::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]

        @test_throws ErrorException one(MMatrix{2,4})
    end

    @testset "eye()" begin
        @test @inferred(eye(SMatrix{2,2,Int})) === @SMatrix [1 0; 0 1]
        @test @inferred(eye(SMatrix{2,2})) === @SMatrix [1.0 0.0; 0.0 1.0]
        @test @inferred(eye(SMatrix{2})) === @SMatrix [1.0 0.0; 0.0 1.0]
        @test @inferred(eye(eye(SMatrix{2,2,Int}))) === @SMatrix [1 0; 0 1]

        @test @inferred(eye(SMatrix{2,3,Int})) === @SMatrix [1 0 0; 0 1 0]
        @test @inferred(eye(SMatrix{2,3})) === @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0]

        @test @inferred(eye(MMatrix{2,2,Int}))::MMatrix == @MMatrix [1 0; 0 1]
        @test @inferred(eye(MMatrix{2,2}))::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]
        @test @inferred(eye(MMatrix{2}))::MMatrix == @MMatrix [1.0 0.0; 0.0 1.0]
    end

    @testset "cross()" begin
        @test @inferred(cross(SVector(1,2,3), SVector(4,5,6))) === SVector(-3, 6, -3)
        @test @inferred(cross(SVector(1,2), SVector(4,5))) === -3
        @test @inferred(cross(SVector(UInt(1),UInt(2)), SVector(UInt(4),UInt(5)))) === -3
        @test @inferred(cross(SVector(UInt(1),UInt(2),UInt(3)), SVector(UInt(4),UInt(5),UInt(6)))) === SVector(-3, 6, -3)
    end

    @testset "inner products" begin
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]

        @test @inferred(dot(v1, v2)) === 40
        @test @inferred(dot(v1, -v2)) === -40
        @test @inferred(dot(v1*im, v2*im)) === 40*im*conj(im)
        @test @inferred(StaticArrays.bilinear_vecdot(v1*im, v2*im)) === 40*im*im
    end

    @testset "transpose() and conj()" begin
        @test @inferred(conj(SVector(1+im, 2+im))) === SVector(1-im, 2-im)

        @test @inferred(transpose(@SVector([1, 2, 3]))) === RowVector(@SVector([1, 2, 3]))
        @test @inferred(transpose(@SMatrix([1 2; 0 3]))) === @SMatrix([1 0; 2 3])
        @test @inferred(transpose(@SMatrix([1 2 3; 4 5 6]))) === @SMatrix([1 4; 2 5; 3 6])

        @test @inferred(adjoint(@SVector([1, 2, 3]))) === RowVector(@SVector([1, 2, 3]))
        @test @inferred(adjoint(@SMatrix([1 2; 0 3]))) === @SMatrix([1 0; 2 3])
        @test @inferred(adjoint(@SMatrix([1 2 3; 4 5 6]))) === @SMatrix([1 4; 2 5; 3 6])
        @test @inferred(adjoint(@SMatrix([1 2*im 3; 4 5 6]))) === @SMatrix([1 4; -2*im 5; 3 6])
    end

    @testset "vcat() and hcat()" begin
        @test @inferred(vcat(SVector(1,2,3))) === SVector(1,2,3)
        @test @inferred(hcat(SVector(1,2,3))) === SMatrix{3,1}(1,2,3)
        @test @inferred(hcat(SMatrix{3,1}(1,2,3))) === SMatrix{3,1}(1,2,3)

        @test @inferred(vcat(SVector(1,2,3), SVector(4,5,6))) === SVector(1,2,3,4,5,6)
        @test @inferred(hcat(SVector(1,2,3), SVector(4,5,6))) === @SMatrix [1 4; 2 5; 3 6]
        @test_throws DimensionMismatch vcat(SVector(1,2,3), @SMatrix [1 4; 2 5])
        @test_throws DimensionMismatch hcat(SVector(1,2,3), SVector(4,5))

        @test @inferred(vcat(@SMatrix([1;2;3]), SVector(4,5,6))) === @SMatrix([1;2;3;4;5;6])
        @test @inferred(vcat(SVector(1,2,3), @SMatrix([4;5;6]))) === @SMatrix([1;2;3;4;5;6])
        @test @inferred(hcat(@SMatrix([1;2;3]), SVector(4,5,6))) === @SMatrix [1 4; 2 5; 3 6]
        @test @inferred(hcat(SVector(1,2,3), @SMatrix([4;5;6]))) === @SMatrix [1 4; 2 5; 3 6]

        @test @inferred(vcat(@SMatrix([1;2;3]), @SMatrix([4;5;6]))) === @SMatrix([1;2;3;4;5;6])
        @test @inferred(hcat(@SMatrix([1;2;3]), @SMatrix([4;5;6]))) === @SMatrix [1 4; 2 5; 3 6]

        @test @inferred(vcat(SVector(1),SVector(2),SVector(3),SVector(4))) === SVector(1,2,3,4)
        @test @inferred(hcat(SVector(1),SVector(2),SVector(3),SVector(4))) === SMatrix{1,4}(1,2,3,4)

        vcat(SVector(1.0f0), SVector(1.0)) === SVector(1.0, 1.0)
        hcat(SVector(1.0f0), SVector(1.0)) === SMatrix{1,2}(1.0, 1.0)
    end

    @testset "normalization" begin
        @test norm(SVector(1.0,2.0,2.0)) ≈ 3.0
        @test norm(SVector(1.0,2.0,2.0),2) ≈ 3.0
        @test norm(SVector(1.0,2.0,2.0),Inf) ≈ 2.0
        @test norm(SVector(1.0,2.0,2.0),1) ≈ 5.0
        @test norm(SVector(1.0,2.0,0.0),0) ≈ 2.0

        @test vecnorm(SVector(1.0,2.0)) ≈ vecnorm([1.0,2.0])
        @test vecnorm(@SMatrix [1 2; 3 4.0+im]) ≈ vecnorm([1 2; 3 4.0+im])
        sm = @SMatrix [1 2; 3 4.0+im]
        @test vecnorm(sm, 3.) ≈ vecnorm([1 2; 3 4.0+im], 3.)
        @test Base.LinAlg.norm_sqr(SVector(1.0,2.0)) ≈ vecnorm([1.0,2.0])^2

        @test normalize(SVector(1,2,3)) ≈ normalize([1,2,3])
        @test normalize(SVector(1,2,3), 1) ≈ normalize([1,2,3], 1)
        @test normalize!(MVector(1.,2.,3.)) ≈ normalize([1.,2.,3.])
        @test normalize!(MVector(1.,2.,3.), 1) ≈ normalize([1.,2.,3.], 1)
    end

    @testset "trace" begin
        @test trace(@SMatrix [1.0 2.0; 3.0 4.0]) === 5.0
        @test_throws DimensionMismatch trace(@SMatrix ones(5,4))
    end

    @testset "size zero" begin
        @test vecdot(SVector{0, Float64}(()), SVector{0, Float64}(())) === 0.
        @test StaticArrays.bilinear_vecdot(SVector{0, Float64}(()), SVector{0, Float64}(())) === 0.
        @test vecnorm(SVector{0, Float64}(())) === 0.
        @test vecnorm(SVector{0, Float64}(()), 1) === 0.
        @test trace(SMatrix{0,0,Float64}(())) === 0.
    end

    @testset "kron" begin
        @test @inferred(kron(@SMatrix([1 2; 3 4]), @SMatrix([0 1 0; 1 0 1]))) ==
            SMatrix{4,6,Int}([0 1 0 0 2 0;
                              1 0 1 2 0 2;
                              0 3 0 0 4 0;
                              3 0 3 4 0 4])
        @test @inferred(kron(@SMatrix([1 2; 3 4]), @SMatrix([2.0]))) === @SMatrix [2.0 4.0; 6.0 8.0]

        # Output should be heap allocated into a SizedArray when it gets large
        # enough.
        M1 = collect(1:20)
        M2 = collect(20:-1:1)'
        @test @inferred(kron(SMatrix{20,1}(M1),SMatrix{1,20}(M2)))::SizedMatrix{20,20} == kron(M1,M2)
    end
end
