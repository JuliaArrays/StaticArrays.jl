using StaticArrays, Test, LinearAlgebra

@testset "Solving linear system" begin
    @testset "Problem size: $n x $n. Matrix type: $m. Element type: $elty" for n in (1,2,3,4,5,8,15),
            (m, v) in ((SMatrix{n,n}, SVector{n}), (MMatrix{n,n}, MVector{n})),
                elty in (Float64, Int)

        A = elty.(rand(-99:2:99, n, n))
        b = A * elty.(rand(2:5, n))
        @test m(A)\v(b) ≈ A\b
    end

    m1 = SMatrix{5,5}(1.0I)
    m2 = SMatrix{2,2}(1.0I)
    v = @SVector ones(4)
    @test_throws DimensionMismatch m1\v
    @test_throws DimensionMismatch m1\m2
end

@testset "Solving linear system (multiple RHS)" begin
    @testset "Problem size: $n x $n. Matrix type: $m1. Element type: $elty" for n in (1,2,3,4,5,8,15),
            (m1, m2) in ((SMatrix{n,n}, SMatrix{n,2}), (MMatrix{n,n}, MMatrix{n,2})),
                elty in (Float64, Int)

        A = elty.(rand(-99:2:99, n, n))
        b = A * elty.(rand(2:5, n, 2))
        @test m1(A)\m2(b) ≈ A\b
    end
end
