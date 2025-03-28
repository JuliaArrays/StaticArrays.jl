using StaticArrays, Test, LinearAlgebra, Random

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


@testset "#1192 The following functions are available for the QR objects: inv, size, and \\." begin
    @testset "pivot=$pivot" for pivot in [Val(true), Val(false)] #, ColumnNorm()]
        y = @SVector rand(5)
        Y = @SMatrix rand(5,5)
        A = @SMatrix rand(5,5)
        A_over = @SMatrix rand(5,6)
        A_under = @SMatrix rand(5,4)

        F = qr(A, pivot)
        F_over = qr(A_over, pivot)
        F_under = qr(A_under, pivot)

        @testset "size" begin
            @test size(A) == (5,5)
            @test size(A_over) == (5,6)
            @test size(A_under) == (5,4)
        end

        @testset "square inversion" begin
            A_inv = inv(F)
            @test inv(F) * A ≈ I(5)
            @test inv(F) ≈ inv(qr(Matrix(A)))
            @test_throws DimensionMismatch inv(F_under)
            @test_throws DimensionMismatch inv(F_over)
        end

        @testset "solve linear system" begin
            x = Matrix(A) \ Vector(y)
            @test x ≈ A \ y ≈ F \ y ≈ F \ Vector(y)

            x_under = Matrix(A_under) \ Vector(y)
            @test x_under == A_under \ y
            @test x_under ≈ F_under \ y
            @test F_under \ y == F_under \ Vector(y)

            x_over = Matrix(A_over) \ Vector(y)
            @test x_over ≈ A_over \ y
            @test A_over * x_over ≈ y

            @test_throws DimensionMismatch F_over \ y
            @test_throws DimensionMismatch qr(Matrix(A_over)) \ y
        end
        
        @testset "solve several linear systems" begin
            @test F \ Y ≈ A \ Y
            @test F_under \ Y ≈ A_under \ Y
        end

        @testset "ldiv!" begin
            x = @MVector zeros(5)
            ldiv!(x, F, y)
            @test x ≈ A \ y

            X = @MMatrix zeros(5,5)
            Y = @SMatrix rand(5,5)
            ldiv!(X, F, Y)
            @test X ≈ A \ Y
        end
        
        @testset "invperm" begin
            x = @SVector [10,15,3,7]
            p = @SVector [4,2,1,3]
            @test x == x[p][invperm(p)]
            @test StaticArrays.is_identity_perm(p[invperm(p)])
            @test_throws Union{BoundsError,ArgumentError} invperm(x)
        end

        @testset "10x faster" begin
            time_to_test = @elapsed (function()
                y2 = @SVector rand(50)
                A2 = @SMatrix rand(50,5)
                F2 = qr(A2, pivot)

                min_time_to_solve = minimum(@elapsed(A2 \ y2) for _ in 1:1_000)
                min_time_to_solve_qr = minimum(@elapsed(F2 \ y2) for _ in 1:1_000)
                @test 10min_time_to_solve_qr < min_time_to_solve
            end)()
            @test time_to_test < 10
        end
    end
end
