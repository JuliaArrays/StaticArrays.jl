using StaticArrays, Compat.Test

@testset "Solving linear system" begin
    @testset "Problem size: $n x $n. Matrix type: $m. Element type: $elty" for n in (1,2,3,4),
            (m, v) in ((SMatrix{n,n}, SVector{n}), (MMatrix{n,n}, MVector{n})),
                elty in (Float64, Int)

        A = elty.(rand(-99:2:99, n, n))
        b = A * elty.(rand(2:5, n))
        @test m(A)\v(b) ≈ A\b
    end

    m1 = @SMatrix eye(5)
    m2 = @SMatrix eye(2)
    v = @SVector ones(4)
    @test_throws DimensionMismatch m1\v
    @test_throws DimensionMismatch m1\m2
end
