using StaticArrays, Test, LinearAlgebra, Random

macro test_noalloc(ex)
    esc(quote
        $ex
        @test(@allocated($ex) == 0)
    end)
end

broadenrandn(::Type{BigFloat}) = BigFloat(randn(Float64))
broadenrandn(::Type{Int}) = rand(-9:9)
broadenrandn(::Type{Complex{T}}) where T = Complex{T}(broadenrandn(T), broadenrandn(T))
broadenrandn(::Type{T}) where T = randn(T)

Random.seed!(42)
@testset "QR decomposition" begin
    function test_qr(arr)

        T = eltype(arr)

        QR = @inferred qr(arr)
        @test QR isa StaticArrays.QR
        Q, R = QR # destructing via iteration
        @test Q isa StaticMatrix
        @test R isa StaticMatrix
        @test eltype(Q) == eltype(R) == typeof((one(T)*zero(T) + zero(T))/norm([one(T)]))

        Q_ref, R_ref = qr(Matrix(arr))
        @test abs.(Q) ≈ abs.(Matrix(Q_ref)) # QR is unique up to diag(Q) signs
        @test abs.(R) ≈ abs.(R_ref)
        @test Q*R ≈ arr
        @test Q'*Q ≈ one(Q'*Q)
        @test istriu(R)

        # pivot=true cases has no StaticArrays specific version yet
        # but fallbacks to LAPACK
        pivot = isdefined(LinearAlgebra, :PivotingStrategy) ? ColumnNorm() : Val(true)
        QRp = @inferred qr(arr, pivot)
        @test QRp isa StaticArrays.QR
        Q, R, p = QRp
        @test Q isa StaticMatrix
        @test R isa StaticMatrix
        @test p isa StaticVector

        Q_ref, R_ref, p_ref = qr(Matrix(arr), pivot)
        @test Q ≈ Matrix(Q_ref)
        @test R ≈ R_ref
        @test p == p_ref
    end

    for eltya in (Float32, Float64, BigFloat, Int),
            rel in (real, complex),
                sz in [(3,3), (3,4), (4,3)]
        arr = SMatrix{sz[1], sz[2], rel(eltya), sz[1]*sz[2]}( [broadenrandn(rel(eltya)) for i = 1:sz[1], j = 1:sz[2]] )
        test_qr(arr)
    end
    # some special cases
    for arr in (
                   (@MMatrix randn(3,2)),
                   (@MMatrix randn(2,3)),
                   (@SMatrix([0 1 2; 0 2 3; 0 3 4; 0 4 5])),
                   (@SMatrix zeros(Int,4,4)),
                   (@SMatrix([1//2 1//1])),
                   (@SMatrix randn(17,18)),    # fallback to LAPACK
                   (@SMatrix randn(18,17))
                )
        test_qr(arr)
    end

    if isdefined(LinearAlgebra, :PivotingStrategy)
        for N = (3, 18)
            A = (@SMatrix randn(N,N))
            @test qr(A, Val(false)) == qr(A, NoPivot())
            @test qr(A, Val(true)) == qr(A, ColumnNorm())
        end
    end
end


@testset "QR method ambiguity" begin
    # Issue #931; just test that methods do not throw an ambiguity error when called
    A = @SMatrix [1.0 2.0 3.0; 4.0 5.0 6.0]
    @test isa(qr(A),              StaticArrays.QR)
    @test isa(qr(A, Val(true)),   StaticArrays.QR)
    @test isa(qr(A, Val(false)),  StaticArrays.QR)
end

@testset "invperm" begin
    p = @SVector [9,8,7,6,5,4,2,1,3]
    v = @SVector [15,14,13,12,11,10,15,3,7]
    @test StaticArrays.is_identity_perm(p[invperm(p)])
    @test v == v[p][invperm(p)]
    @test_throws ArgumentError invperm(v)
    expect0 = Base.JLOptions().check_bounds != 1
    # expect0 && @test_noalloc @inbounds invperm(p)

    @test StaticArrays.invpivot(v, p) == v[invperm(p)]
    @test_noalloc StaticArrays.invpivot(v, p)
end

@testset "#1192 QR inv, size, and \\" begin
    function test_pivot(pivot, MatrixType)
        Random.seed!(42)
        A = rand(MatrixType)
        n, m = size(A)
        y = @SVector rand(size(A, 1))
        Y = @SMatrix rand(n, 2)
        F = @inferred QR qr(A, pivot)
        F_gold = @inferred LinearAlgebra.QRCompactWY qr(Matrix(A), pivot)

        expect0 = pivot isa NoPivot || Base.JLOptions().check_bounds != 1

        @test StaticArrays.is_identity_perm(F.p) == (pivot isa NoPivot)
        @test size(F) == size(A)

        @testset "inv UpperTriangular StaticMatrix" begin
            if m <= n
                invR = @inferred StaticMatrix inv(UpperTriangular(F.R))
                @test invR*F.R ≈ I(m)

                expect0 && @eval @test_noalloc inv(UpperTriangular($F.R))
            else
                @test_throws DimensionMismatch inv(UpperTriangular(F.R))
            end
        end

        @testset "qr inversion" begin
            if m <= n
                inv_F_gold = inv(qr(Matrix(A)))
                inv_F = @inferred StaticMatrix inv(F)
                @test size(inv_F) == size(inv_F_gold)
                @test inv_F[1:m,:] ≈ inv_F_gold[1:m,:] # equal except for the nullspace
                @test inv_F * A ≈ I(n)[:,1:m]

                expect0 && @eval @test_noalloc inv($F)
            else
                @test_throws DimensionMismatch inv(F)
                @test_throws DimensionMismatch inv(qr(Matrix(A)))
            end
        end

        @testset "QR \\ StaticVector" begin
            if m <= n
                x_gold = Matrix(A) \ Vector(y)
                x = @inferred StaticVector F \ y
                @test x_gold ≈ x

                expect0 && @eval @test_noalloc $F \ $y
            else
                @test_throws DimensionMismatch F \ y

                if pivot isa Val{false}
                    @test_throws DimensionMismatch F_gold \ Vector(y)
                end
            end
        end

        @testset "QR \\ StaticMatrix" begin
            if m <= n
                @test F \ Y ≈ A \ Y

                expect0 && @eval @test_noalloc $F \ $Y
            else
                @test_throws DimensionMismatch F \ Y
            end
        end

        @testset "ldiv!" begin
            x = @MVector zeros(m)
            X = @MMatrix zeros(m, size(Y, 2))

            if m <= n
                ldiv!(x, F, y)
                @test x ≈ A \ y

                ldiv!(X, F, Y)
                @test X ≈ A \ Y

                expect0 && @test_noalloc ldiv!(x, F, y)
                expect0 && @test_noalloc ldiv!(X, F, Y)
            else
                @test_throws DimensionMismatch ldiv!(x, F, y)
                @test_throws DimensionMismatch ldiv!(X, F, Y)

                if pivot isa Val{false}
                    @test_throws DimensionMismatch ldiv!(zeros(size(x)), F_gold, Array(y))
                    @test_throws DimensionMismatch ldiv!(zeros(size(X)), F_gold, Array(Y))
                end
            end
        end
    end

    @testset "pivot=$pivot" for pivot in [NoPivot(), ColumnNorm()]
        @testset "$label ($n,$m)" for (label,n,m) in [
                (:square,3,3),
                (:overdetermined,6,3),
                (:underdetermined,3,4)
                ]
            test_pivot(pivot, SMatrix{n,m,Float64})
        end

        @testset "performance" begin
            function speed_test(n, iter)
                y2 = @SVector rand(n)
                A2 = @SMatrix rand(n,5)
                F2 = qr(A2, pivot)
                iA = pinv(A2)

                min_time_to_solve = minimum(@elapsed(A2 \ y2) for _ in 1:iter)
                min_time_to_solve_qr = minimum(@elapsed(F2 \ y2) for _ in 1:iter)
                min_time_to_solve_inv = minimum(@elapsed(iA * y2) for _ in 1:iter)

                if 1 != Base.JLOptions().check_bounds
                    @test 10min_time_to_solve_qr < min_time_to_solve
                    @test 2min_time_to_solve_inv < min_time_to_solve_qr
                end
            end
            speed_test(100, 100)
            @test @elapsed(speed_test(100, 100)) < 1
        end
    end
end
