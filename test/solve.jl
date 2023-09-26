using StaticArrays, Test, LinearAlgebra

@testset "Solving linear system" begin
    @testset "Problem size: $n x $n. Matrix type: $m. Element type: $elty, Wrapper: $wrapper" for n in (1,2,3,4,5,8,15),
            (m, v) in ((SMatrix{n,n}, SVector{n}), (MMatrix{n,n}, MVector{n})),
                elty in (Float64, Int, BigFloat), wrapper in (identity, Symmetric, Hermitian)

        A = wrapper(elty.(rand(-99:2:99, n, n)))
        b = A * elty.(rand(2:5, n))
        @test wrapper(m(A))\v(b) ≈ wrapper(A)\b
    end

    m1 = SMatrix{5,5}(1.0I)
    m2 = SMatrix{2,2}(1.0I)
    v = @SVector ones(4)
    @test_throws DimensionMismatch m1\v
    @test_throws DimensionMismatch m1\m2

    # Mixed static/dynamic arrays
    # \ specializes for Diagonal, LowerTriangular, UpperTriangular, square, and non-square
    # So try all of these
    @testset "Mixed static/dynamic" begin
        v = @SVector([0.2,0.3])
        for m in (@SMatrix([1.0 0; 0 1.0]), @SMatrix([1.0 0; 1.0 1.0]),
                  @SMatrix([1.0 1.0; 0 1.0]), @SMatrix([1.0 0.5; 0.25 1.0]))
            # TODO: include @SMatrix([1.0 0.0 0.0; 1.0 2.0 0.5]), need qr methods
            @test m \ v ≈ Array(m) \ v ≈ m \ Array(v) ≈ Array(m) \ Array(v)
        end
    end
end


@testset "Solving linear system (multiple RHS)" begin
    @testset "Problem size: $n x $n. Matrix type: $m1. Element type: $elty" for n in (1,2,3,4,5,8,15),
            (m1, m2) in ((SMatrix{n,n}, SMatrix{n,2}), (MMatrix{n,n}, MMatrix{n,2})),
                elty in (Float64, Int, BigFloat)

        A = elty.(rand(-99:2:99, n, n))
        b = A * elty.(rand(2:5, n, 2))
        @test m1(A)\m2(b) ≈ A\b

    end

    # Solve with non-square left hand sides (#606)
    m1 = @SMatrix[0.2 0.3
                  0.0 0.1
                  0.5 0.1]
    m2 = @SVector[1,2,3]
    @test @inferred(m1\m2) ≈ Array(m1)\Array(m2)
    m2 = @SMatrix[1 4
                  2 5
                  3 6]
    @test @inferred(m1\m2) ≈ Array(m1)\Array(m2)

    @testset "Mixed static/dynamic" begin
        m2 = @SMatrix([0.2 0.3; 0.0 0.1])
        for m1 in (@SMatrix([1.0 0; 0 1.0]), @SMatrix([1.0 0; 1.0 1.0]),
                   @SMatrix([1.0 1.0; 0 1.0]), @SMatrix([1.0 0.5; 0.25 1.0]))
            # TODO: include @SMatrix([1.0 0.0 0.0; 1.0 2.0 0.5]), need qr methods
            @test m1 \ m2 ≈ Array(m1) \ m2 ≈ m1 \ Array(m2) ≈ Array(m1) \ Array(m2)
        end
    end
end
