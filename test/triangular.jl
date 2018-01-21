@testset "Triangular-matrix multiplication" begin
    for n in (1, 2, 3, 4),
        eltyA in (Float64, Complex128, Int),
            (t, uplo) in ((UpperTriangular, :U), (LowerTriangular, :L)),
                eltyB in (Float64, Complex128)

        A = t(eltyA == Int ? rand(1:7, n, n) : convert(Matrix{eltyA}, (eltyA <: Complex ? complex.(randn(n, n), randn(n, n)) : randn(n, n)) |> t -> chol(t't) |> t -> uplo == :U ? t : adjoint(t)))
        B = convert(Matrix{eltyB}, eltyA <: Complex ? real(A)*ones(n, n) : A*ones(n, n))
        SA = t(SMatrix{n,n}(A.data))
        SB = SMatrix{n,n}(B)

        @test (SA*SB[:,1])::SVector{n} ≈ A*B[:,1]
        @test (SA*SB)::SMatrix{n,n} ≈ A*B
        @test (SA*SB.')::SMatrix{n,n} ≈ A*B.'
        @test (SA*SB')::SMatrix{n,n} ≈ A*B'
        @test (SA'*SB[:,1])::SVector{n} ≈ A'*B[:,1]
        @test (SA'*SB)::SMatrix{n,n} ≈ A'*B
        @test (SA'*SB')::SMatrix{n,n} ≈ A'*B'
        @test (SA.'*SB[:,1])::SVector{n} ≈ A.'*B[:,1]
        @test (SA.'*SB)::SMatrix{n,n} ≈ A.'*B
        @test (SA.'*SB.')::SMatrix{n,n} ≈ A.'*B.'
        @test (SB*SA)::SMatrix{n,n} ≈ B*A
        @test (SB[:,1].'*SA)::RowVector{<:Any,<:SVector{n}} ≈ B[:,1].'*A
        @test (SB.'*SA)::SMatrix{n,n} ≈ B.'*A
        @test SB[:,1]'*SA ≈ B[:,1]'*A
        @test (SB'*SA)::SMatrix{n,n} ≈ B'*A
        @test (SB*SA')::SMatrix{n,n} ≈ B*A'
        @test (SB*SA.')::SMatrix{n,n} ≈ B*A.'
        @test (SB[:,1].'*SA.')::RowVector{<:Any,<:SVector{n}} ≈ B[:,1].'*A.'
        @test (SB.'*SA.')::SMatrix{n,n} ≈ B.'*A.'
        @test (SB[:,1]'*SA') ≈ SB[:,1]'*SA'
        @test (SB'*SA')::SMatrix{n,n} ≈ B'*A'

        @test_throws DimensionMismatch SA*ones(SVector{n+1,eltyB})
        @test_throws DimensionMismatch ones(SMatrix{n+1,n+1,eltyB})*SA
        @test_throws DimensionMismatch SA.'*ones(SVector{n+1,eltyB})
        @test_throws DimensionMismatch SA'*ones(SVector{n+1,eltyB})
        @test_throws DimensionMismatch ones(SMatrix{n+1,n+1,eltyB})*SA.'
        @test_throws DimensionMismatch ones(SMatrix{n+1,n+1,eltyB})*SA'
    end
end

@testset "Triangular-RowVector multiplication" begin
    for n in (1,),
        eltyA in (Float64, Complex128, Int),
            (t, uplo) in ((UpperTriangular, :U), (LowerTriangular, :L)),
                eltyB in (Float64, Complex128)

        A = t(eltyA == Int ? rand(1:7, n, n) : convert(Matrix{eltyA}, (eltyA <: Complex ? complex.(randn(n, n), randn(n, n)) : randn(n, n)) |> t -> chol(t't) |> t -> uplo == :U ? t : adjoint(t)))
        B = convert(Vector{eltyB}, eltyA <: Complex ? real(A)*ones(n) : A*ones(n))
        SA = t(SMatrix{n,n}(A.data))
        SB = SVector{n}(B).'

        @test (SA*SB)::Matrix ≈ A*B.'
        @test (SA*SB.')::SVector{n} ≈ A*B
        # Some weird type stuff going on here
        @test (SA*SB') ≈ A*conj(B)
        @test (SA'*SB)::Matrix ≈ A'*B.'
        @test (SA'*SB.')::SVector{n} ≈ A'*B
        @test (SA'*SB') ≈ A'*conj(B)
        @test (SA.'*SB)::Matrix ≈ A.'*B.'
        @test (SA.'*SB.')::SVector{n} ≈ A.'*B
        @test (SA.'*SB') ≈ A.'*conj(B)
        @test (SB*SA)::RowVector{<:Any,<:SVector{n}} ≈ B.'*A
        @test (SB*SA') ≈ B.'*A'
        @test (SB*SA.')::RowVector{<:Any,<:SVector{n}} ≈ B.'*A.'
        @test (SB.'*SA)::Matrix ≈ B*A
        @test (SB'*SA)::Matrix ≈ conj(B)*A
        @test (SB.'*SA.')::Matrix ≈ B*A.'
        @test (SB.'*SA')::Matrix ≈ B*A'
        @test (SB'*SA.')::Matrix ≈ conj(B)*A.'
        @test (SB'*SA')::Matrix ≈ conj(B)*A'
    end
end

@testset "Triangular-matrix division" begin
    for n in (1, 2, 3, 4),
        eltyA in (Float64, Complex128, Int),
            (t, uplo) in ((UpperTriangular, :U), (LowerTriangular, :L)),
                eltyB in (Float64, Complex128)

        A = t(eltyA == Int ? rand(1:7, n, n) : convert(Matrix{eltyA}, (eltyA <: Complex ? complex.(randn(n, n), randn(n, n)) : randn(n, n)) |> t -> chol(t't) |> t -> uplo == :U ? t : adjoint(t)))
        B = convert(Matrix{eltyB}, eltyA <: Complex ? real(A)*ones(n, n) : A*ones(n, n))
        SA = t(SMatrix{n,n}(A.data))
        SB = SMatrix{n,n}(B)

        @test (SA\SB[:,1])::SVector{n} ≈ A\B[:,1]
        @test (SA\SB)::SMatrix{n,n} ≈ A\B
        @test (SA.'\SB[:,1])::SVector{n} ≈ A.'\B[:,1]
        @test (SA.'\SB)::SMatrix{n,n} ≈ A.'\B
        @test (SA'\SB[:,1])::SVector{n} ≈ A'\B[:,1]
        @test (SA'\SB)::SMatrix{n,n} ≈ A'\B

        @test_throws DimensionMismatch SA\ones(SVector{n+2,eltyB})
        @test_throws DimensionMismatch SA.'\ones(SVector{n+2,eltyB})
        @test_throws DimensionMismatch SA'\ones(SVector{n+2,eltyB})

        @test_throws LinearAlgebra.SingularException t(zeros(SMatrix{n,n,eltyA}))\ones(SVector{n,eltyB})
        @test_throws LinearAlgebra.SingularException t(zeros(SMatrix{n,n,eltyA})).'\ones(SVector{n,eltyB})
        @test_throws LinearAlgebra.SingularException t(zeros(SMatrix{n,n,eltyA}))'\ones(SVector{n,eltyB})
    end
end
