using StaticArrays, Test, LinearAlgebra

@testset "Linear algebra" begin

    @testset "SArray as a (mathematical) vector space" begin
        c = 2
        v1 = @SVector [2,4,6,8]
        v2 = @SVector [4,3,2,1]

        @test @inferred(v1 * c) === @SVector [4,8,12,16]
        @test @inferred(v1 / c) === @SVector [1.0,2.0,3.0,4.0]
        @test @inferred(c \ v1)::SVector ≈ @SVector [1.0,2.0,3.0,4.0]

        @test @inferred(v1 + v2) === @SVector [6, 7, 8, 9]
        @test @inferred(v1 - v2) === @SVector [-2, 1, 4, 7]

        # #528 eltype with empty addition
        zm = zeros(SMatrix{3, 0, Float64})
        @test @inferred(zm + zm) === zm

        # TODO Decide what to do about this stuff:
        #v3 = [2,4,6,8]
        #v4 = [4,3,2,1]

        #@test @inferred(v1 + v4) === @SVector [6, 7, 8, 9]
        #@test @inferred(v3 + v2) === @SVector [6, 7, 8, 9]
        #@test @inferred(v1 - v4) === @SVector [-2, 1, 4, 7]
        #@test @inferred(v3 - v2) === @SVector [-2, 1, 4, 7]
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

    @testset "Constructors from UniformScaling" begin
        I3x3 = Matrix(I, 3, 3)
        I3x2 = Matrix(I, 3, 2)
        # SArray
        ## eltype from I
        @test @inferred(SArray{Tuple{3,3}}(I))::SMatrix{3,3,Bool,9} == I3x3
        @test @inferred(SArray{Tuple{3,3}}(2.0I))::SMatrix{3,3,Float64,9} == 2I3x3
        ## eltype from constructor
        @test @inferred(SArray{Tuple{3,3},Float64}(I))::SMatrix{3,3,Float64,9} == I3x3
        @test @inferred(SArray{Tuple{3,3},Float32}(2.0I))::SMatrix{3,3,Float32,9} == 2I3x3
        ## non-square
        @test @inferred(SArray{Tuple{3,2}}(I))::SMatrix{3,2,Bool,6} == I3x2
        # SMatrix
        @test @inferred(SMatrix{3,3}(I))::SMatrix{3,3,Bool,9} == I3x3
        @test @inferred(SMatrix{3,3}(2.0I))::SMatrix{3,3,Float64,9} == 2I3x3
        # MArray
        ## eltype from I
        @test @inferred(MArray{Tuple{3,3}}(I))::MMatrix{3,3,Bool,9} == I3x3
        @test @inferred(MArray{Tuple{3,3}}(2.0I))::MMatrix{3,3,Float64,9} == 2I3x3
        ## eltype from constructor
        @test @inferred(MArray{Tuple{3,3},Float64}(I))::MMatrix{3,3,Float64,9} == I3x3
        @test @inferred(MArray{Tuple{3,3},Float32}(2.0I))::MMatrix{3,3,Float32,9} == 2I3x3
        ## non-square
        @test @inferred(MArray{Tuple{3,2}}(I))::MMatrix{3,2,Bool,6} == I3x2
        # MMatrix
        @test @inferred(MMatrix{3,3}(I))::MMatrix{3,3,Bool,9} == I3x3
        @test @inferred(MMatrix{3,3}(2.0I))::MMatrix{3,3,Float64,9} == 2I3x3
        # SDiagonal
        @test @inferred(SDiagonal{3}(I))::SDiagonal{3,Bool} == I3x3
        @test @inferred(SDiagonal{3}(2.0I))::SDiagonal{3,Float64} == 2I3x3
        @test @inferred(SDiagonal{3,Float64}(I))::SDiagonal{3,Float64} == I3x3
        @test @inferred(SDiagonal{3,Float32}(2.0I))::SDiagonal{3,Float32} == 2I3x3
    end

    @testset "diagm()" begin
        @test @inferred(diagm(Val(0) => SVector(1,2))) === @SMatrix [1 0; 0 2]
        @test @inferred(diagm(Val(2) => SVector(1,2,3)))::SMatrix == diagm(2 => [1,2,3])
        @test @inferred(diagm(Val(-2) => SVector(1,2,3)))::SMatrix == diagm(-2 => [1,2,3])
        @test @inferred(diagm(Val(-2) => SVector(1,2,3), Val(1) => SVector(4,5)))::SMatrix == diagm(-2 => [1,2,3], 1 => [4,5])
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
        # inner product, whether via `dot` or written out as `x'*y`, should be recursive like Base:
        @test @inferred(dot(@SVector[[1,2],[3,4]], @SVector[[1,2],[3,4]])) === 30
        @test @inferred(@SVector[[1,2],[3,4]]' * @SVector[[1,2],[3,4]]) === 30

        m1 = reshape(v1, Size(2,2))
        m2 = reshape(v2, Size(2,2))
        @test @inferred(dot(m1, m2)) === dot(v1, v2)
    end

    @testset "transpose() and conj()" begin
        @test @inferred(conj(SVector(1+im, 2+im))) === SVector(1-im, 2-im)

        @test @inferred(transpose(@SVector([1, 2, 3]))) === Transpose(@SVector([1, 2, 3]))
        @test @inferred(adjoint(@SVector([1, 2, 3]))) === Adjoint(@SVector([1, 2, 3]))
        @test @inferred(transpose(@SMatrix([1 2; 0 3]))) === @SMatrix([1 0; 2 3])
        @test @inferred(transpose(@SMatrix([1 2 3; 4 5 6]))) === @SMatrix([1 4; 2 5; 3 6])

        @test @inferred(adjoint(@SMatrix([1 2; 0 3]))) === @SMatrix([1 0; 2 3])
        @test @inferred(adjoint(@SMatrix([1 2 3; 4 5 6]))) === @SMatrix([1 4; 2 5; 3 6])
        @test @inferred(adjoint(@SMatrix([1 2*im 3; 4 5 6]))) === @SMatrix([1 4; -2*im 5; 3 6])

        m = [1 2; 3 4] + im*[5 6; 7 8]
        @test @inferred(adjoint(@SVector [m,m])) == adjoint([m,m])
        @test @inferred(transpose(@SVector [m,m])) == transpose([m,m])
        @test @inferred(adjoint(@SMatrix [m m; m m])) == adjoint([[m] [m]; [m] [m]])
        @test @inferred(transpose(@SMatrix [m m; m m])) == transpose([[m] [m]; [m] [m]])

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

        # issue #388
        let x = SVector(1, 2, 3)
            # current limit: 34 arguments
            hcat(
                x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x)
            allocs = @allocated hcat(
                x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x)
            @test allocs == 0
            vcat(
                x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x)
            allocs = @allocated vcat(
                x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x)
            @test allocs == 0
        end

        # issue #561
        let A = Diagonal(SVector(1, 2)), B = @SMatrix [3 4; 5 6]
            @test @inferred(hcat(A, B)) === SMatrix{2, 4}([Matrix(A) Matrix(B)])
        end

        let A = Transpose(@SMatrix [1 2; 3 4]), B = Adjoint(@SMatrix [5 6; 7 8])
            @test @inferred(hcat(A, B)) === SMatrix{2, 4}([Matrix(A) Matrix(B)])
        end

        let A = Diagonal(SVector(1, 2)), B = @SMatrix [3 4; 5 6]
            @test @inferred(vcat(A, B)) === SMatrix{4, 2}([Matrix(A); Matrix(B)])
        end

        let A = Transpose(@SMatrix [1 2; 3 4]), B = Adjoint(@SMatrix [5 6; 7 8])
            @test @inferred(vcat(A, B)) === SMatrix{4, 2}([Matrix(A); Matrix(B)])
        end
    end

    @testset "normalization" begin
        @test norm(SVector(1.0,2.0,2.0)) ≈ 3.0
        @test norm(SVector(1.0,2.0,2.0),2) ≈ 3.0
        @test norm(SVector(1.0,2.0,2.0),Inf) ≈ 2.0
        @test norm(SVector(1.0,2.0,2.0),1) ≈ 5.0
        @test norm(SVector(1.0,2.0,0.0),0) ≈ 2.0

        @test norm(SVector(1.0,2.0)) ≈ norm([1.0,2.0])
        @test norm(@SMatrix [1 2; 3 4.0+im]) ≈ norm([1 2; 3 4.0+im])
        sm = @SMatrix [1 2; 3 4.0+im]
        @test norm(sm, 3.) ≈ norm([1 2; 3 4.0+im], 3.)
        @test LinearAlgebra.norm_sqr(SVector(1.0,2.0)) ≈ norm([1.0,2.0])^2

        @test normalize(SVector(1,2,3)) ≈ normalize([1,2,3])
        @test normalize(SVector(1,2,3), 1) ≈ normalize([1,2,3], 1)
        @test normalize!(MVector(1.,2.,3.)) ≈ normalize([1.,2.,3.])
        @test normalize!(MVector(1.,2.,3.), 1) ≈ normalize([1.,2.,3.], 1)
    end

    @testset "trace" begin
        @test tr(@SMatrix [1.0 2.0; 3.0 4.0]) === 5.0
        @test_throws DimensionMismatch tr(@SMatrix ones(5,4))
    end

    @testset "size zero" begin
        @test dot(SVector{0, Float64}(()), SVector{0, Float64}(())) === 0.
        @test StaticArrays.bilinear_vecdot(SVector{0, Float64}(()), SVector{0, Float64}(())) === 0.
        @test norm(SVector{0, Float64}(())) === 0.
        @test norm(SVector{0, Float64}(()), 1) === 0.
        @test tr(SMatrix{0,0,Float64}(())) === 0.
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

        test_kron = function (a, b, A, p, q, P)
            @test @inferred(kron(a,b)) ===  SVector{9}(kron(p,q))
            @test @inferred(kron(b,a)) ===  SVector{9}(kron(q,p))
            @test @inferred(kron(a',b')) ===  SMatrix{1,9}(kron(p',q'))
            @test @inferred(kron(b',a')) ===  SMatrix{9,1}(kron(q',p'))'
            @test @inferred(kron(b',a)) ===  SMatrix{3,3}(kron(q',p))
            @test @inferred(kron(b,a')) ===  SMatrix{3,3}(kron(q,p'))
            @test @inferred(kron(b,A)) ===  SMatrix{6,2}(kron(q,P))
            @test @inferred(kron(b',A)) ===  SMatrix{2,6}(kron(q',P))
            @test @inferred(kron(A,b)) ===  SMatrix{6,2}(kron(P,q))
            @test @inferred(kron(A,b')) ===  SMatrix{2,6}(kron(P,q'))

            @test @inferred(kron(transpose(a),transpose(b))) ===  SMatrix{1,9}(kron(transpose(p),transpose(q)))
            @test @inferred(kron(transpose(b),transpose(a))) ===  SMatrix{9,1}(kron(transpose(q),transpose(p)))'
            @test @inferred(kron(transpose(b),a)) ===  SMatrix{3,3}(kron(transpose(q),p))
            @test @inferred(kron(b,transpose(a))) ===  SMatrix{3,3}(kron(q,transpose(p)))
            @test @inferred(kron(transpose(b),A)) ===  SMatrix{2,6}(kron(transpose(q),P))
            @test @inferred(kron(A,transpose(b))) ===  SMatrix{2,6}(kron(P,transpose(q)))
        end

        # Tests for kron of two SVectors as well as between SVectors and SMatrices.
        a = @SVector [1, 2, 3]
        b = @SVector [4, 5, 6]
        A = @SMatrix [1 2; 3 4]

        p = [1,2,3]
        q = [4,5,6]
        P = [1 2; 3 4]

        test_kron(a, b, A, p, q, P)


        # Tests for kron of two SVectors as well as between SVectors and
        # SMatrices and verifies type promotion (e.g. Int to Float).
        a = @SVector [1, 2, 3]
        b = @SVector [4.5, 5.5, 6.5]
        A = @SMatrix [1 2; 3 4]

        p = [1,2,3]
        q = [4.5, 5.5, 6.5]
        P = [1 2; 3 4]

        test_kron(a, b, A, p, q, P)

        # Output should be heap allocated into a SizedArray when it gets large
        # enough.
        p = collect(1:10)
        q = collect(1:21)
        P = randn(10,10)
        a = SVector{10}(p)
        b = SVector{21}(q)
        A = SMatrix{10,10}(P)

        @test @inferred(kron(a,b))::SizedVector{210} == kron(p,q)
        @test @inferred(kron(b,a))::SizedVector{210} == kron(q,p)
        @test @inferred(kron(b,a'))::SizedMatrix{21,10} == kron(q,p')
        @test @inferred(kron(b',a))::SizedMatrix{10,21} == kron(q',p)
        @test @inferred(kron(a',b'))::SizedMatrix{1,210} == kron(p',q')
        @test @inferred(kron(b',a'))::SizedMatrix{1,210} == kron(q',p')
        @test @inferred(kron(b',A))::SizedMatrix{10,210} == kron(q',P)
        @test @inferred(kron(A,b'))::SizedMatrix{10,210} == kron(P,q')
        @test @inferred(kron(b,A))::SizedMatrix{210,10} == kron(q,P)
        @test @inferred(kron(A,b))::SizedMatrix{210,10} == kron(P,q)

        @test @inferred(kron(b,transpose(a)))::SizedMatrix{21,10} == kron(q,transpose(p))
        @test @inferred(kron(transpose(b),a))::SizedMatrix{10,21} == kron(transpose(q),p)
        @test @inferred(kron(transpose(a),transpose(b)))::SizedMatrix{1,210} == kron(transpose(p),transpose(q))
        @test @inferred(kron(transpose(b),transpose(a)))::SizedMatrix{1,210} == kron(transpose(q),transpose(p))
        @test @inferred(kron(transpose(b),A))::SizedMatrix{10,210} == kron(transpose(q),P)
        @test @inferred(kron(A,transpose(b)))::SizedMatrix{10,210} == kron(P,transpose(q))

    end
end
