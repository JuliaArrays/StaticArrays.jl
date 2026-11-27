using StaticArrays, Test, LinearAlgebra

# Mutating factorizations may, but need not, use their input as workspace: for static
# matrices, they should give the same result as their non-mutating counterpart.
@testset "Mutating factorizations" begin
    S  = @SMatrix [2.0 1.0; 1.0 3.0]
    Sc = @SMatrix [2.0 1.0im; -1.0im 3.0]

    @testset "eigen!, eigvals!" begin
        for A in (S, MMatrix(S), Symmetric(S), Hermitian(S), Hermitian(Sc), transpose(S),
                  adjoint(S), Diagonal(SVector(2.0, 3.0)), UpperTriangular(S))
            E = eigen(A)
            E! = eigen!(copy(A))
            @test E!.values == E.values
            @test E!.vectors == E.vectors
            @test eigvals!(copy(A)) == eigvals(A)
        end
        # keyword arguments are passed on
        @test eigen!(copy(S); permute = false).values == eigen(S; permute = false).values
        @test eigvals!(MMatrix(S); scale = false) == eigvals(S; scale = false)
    end

    @testset "svd!, svdvals!" begin
        for A in (S, MMatrix(S), transpose(S), UpperTriangular(S))
            F = svd(A)
            F! = svd!(copy(A))
            @test F!.U == F.U
            @test F!.S == F.S
            @test F!.Vt == F.Vt
            @test svdvals!(copy(A)) == svdvals(A)
        end
        # `Symmetric`/`Hermitian` use a dedicated method (avoids an ambiguity)
        for A in (Symmetric(S), Hermitian(S), Hermitian(Sc))
            @test svdvals!(copy(A)) == svdvals(A)
        end
    end

    @testset "lu!" begin
        for A in (S, Symmetric(S), Hermitian(S), Hermitian(Sc))
            F = lu(A)
            for F! in (lu!(A), lu!(A, Val(true)), lu!(A; check = false))
                @test F!.L == F.L
                @test F!.U == F.U
                @test F!.p == F.p
            end
            @test lu!(A, Val(false)).U == lu(A, Val(false)).U
            @static if isdefined(LinearAlgebra, :PivotingStrategy)
                @test lu!(A, RowMaximum()).U == lu(A, RowMaximum()).U
                @test lu!(A, NoPivot()).U == lu(A, NoPivot()).U
            end
        end
        # mutable static matrices still use LinearAlgebra's in-place algorithm
        M = MMatrix(S)
        @test lu!(M).factors === M
    end

    @testset "qr!" begin
        F = qr(S)
        F! = qr!(S)
        @test F!.Q == F.Q
        @test F!.R == F.R
        @test qr!(S, Val(true)).R == qr(S, Val(true)).R
        @test qr!(S, Val(false)).R == qr(S, Val(false)).R
        @static if isdefined(LinearAlgebra, :PivotingStrategy)
            @test qr!(S, ColumnNorm()).R == qr(S, ColumnNorm()).R
            @test qr!(S, NoPivot()).R == qr(S, NoPivot()).R
        end
    end

    @testset "bunchkaufman!" begin
        for A in (S, Symmetric(S), Hermitian(S), Hermitian(Sc))
            F = bunchkaufman(A)
            F! = bunchkaufman!(A)
            @test F!.D == F.D
            @test F!.p == F.p
            @test Matrix(F!) ≈ A
        end
    end
end
