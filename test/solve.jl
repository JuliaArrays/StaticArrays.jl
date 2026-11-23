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
        # Square matrices
        for m in (@SMatrix([1.0 0; 0 1.0]), @SMatrix([1.0 0; 1.0 1.0]),
                  @SMatrix([1.0 1.0; 0 1.0]), @SMatrix([1.0 0.5; 0.25 1.0]))
            @test m \ v ≈ Array(m) \ v ≈ m \ Array(v) ≈ Array(m) \ Array(v)
        end
        # Rectangular matrices
        for m in (@SMatrix([1.0 0.0 0.0; 1.0 2.0 0.5]), @SMatrix([1.0 2.0 0.5; 0.0 0.0 1.0]),
                  @SMatrix([0.0 0.0 1.0; 1.0 2.0 0.5]), @SMatrix([1.0 2.0 0.5; 1.0 0.0 0.0]))
            @test m \ v ≈ Array(m) \ v ≈ Array(m) \ Array(v)
            @test_throws MethodError m \ Array(v) # TODO: requires adjoint(::QR) method
        end
    end
    @testset "More static tests" begin
        # 1) 3×5 real, two RHS
        A1 = @SMatrix [1.0  2.0  3.0  4.0  5.0;
            0.0  1.0  0.0  1.0  0.0;
            -1.0  0.0  2.0 -2.0  1.0]
        B1 = @SMatrix [1.0 0.0;
            0.0 1.0;
            1.0 1.0]

        # 2) 4×6 real
        A2 = @SMatrix [ 2.0  -1.0  0.0  4.0  1.0  3.0;
            -3.0   2.0  5.0 -1.0  0.0  2.0;
            1.0   0.0  1.0  0.0  2.0 -2.0;
            0.0   3.0 -1.0  1.0  1.0  0.0]
        b2_1 = @SVector [1.0, 4.0, -2.0, 0.5]
        b2_2 = @SMatrix [1.0 1.0
            4.0 6.0
            -2.0 2.0
            0.5 1.5]

        # 3) 3×4 complex
        A3 = @SMatrix [1+2im   0+1im  2-1im  3+0im;
            0+0im   2+0im  1+1im  0-2im;
            3-1im  -1+0im  0+2im  1+0im]
        b3_1 = @SVector [1+0im, 2-1im, -1+3im]
        b3_2 = @SMatrix [
            1+0im -9+0im
            2-1im 2-4im
            -1+3im 2+3im]

        # 4) 3×6 rank-deficient (cols 3 = 1+2, col 4 = col 1, col 5 = col 2, col 6 = zeros)
        A4 = @SMatrix [1.0 2.0 3.0 1.0 2.0 0.0;
            0.0 1.0 1.0 0.0 1.0 0.0;
            1.0 3.0 4.0 1.0 3.0 0.0]
        b4_1 = @SVector [1.0, 0.0, 1.0]
        b4_2 = @SMatrix [1.0 0.0
                    0.0 1.0
                    1.0 0.0]
        for (A, B) in [(A1, B1), (A2, b2_1), (A2, b2_2), (A3, b3_1), (A3, b3_2), (A4, b4_1), (A4, b4_2)]
            @test A \ B ≈ Array(A) \ Array(B)
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
