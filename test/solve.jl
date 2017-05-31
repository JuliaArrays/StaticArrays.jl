@testset "Solving linear system" begin

    # I see an inference error in the combined testset below. I can't reproduce
    # at the REPL or in a function for individual n, etc... speculatively, it
    # might be reusing A, b with different types in the same (non-toplevel)
    # scope, for which I've come accross inference bugs in the past.

    for n in (1,2,3,4),
            (m, v) in ((SMatrix{n,n}, SVector{n}), (MMatrix{n,n}, MVector{n})),
                elty in (Float64, Int)

        eval(quote
            A = $elty.(rand(-99:2:99,$n,$n))
            b = A*ones($elty,$n)
            @test $m(A)\$v(b) ≈ A\b
        end)
    end

    #=@testset "Problem size: $n x $n. Matrix type: $m. Element type: $elty" for n in (1,2,3,4),
            (m, v) in ((SMatrix{n,n}, SVector{n}), (MMatrix{n,n}, MVector{n})),
                elty in (Float64, Int)

        A = elty.(rand(-99:2:99,n,n))
        b = A*ones(elty,n)
        @test m(A)\v(b) ≈ A\b
    end =#

    m1 = @SMatrix eye(5)
    m2 = @SMatrix eye(2)
    v = @SVector ones(4)
    @test_throws DimensionMismatch m1\v
    @test_throws DimensionMismatch m1\m2
end

@testset "Solving triangular system" begin
    for n in (1, 2, 3, 4),
        (t, uplo) in ((UpperTriangular, :U),
                      (LowerTriangular, :L)),
            (m, v, u) in ((SMatrix{n,n}, SVector{n}, SMatrix{n,2}),
                          (MMatrix{n,n}, MVector{n}, SMatrix{n,2})),
                eltya in (Float32, Float64, BigFloat, Complex64, Complex128, Complex{BigFloat}, Int),
                    eltyb in (Float32, Float64, BigFloat, Complex64, Complex128, Complex{BigFloat})

        A = t(eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, (eltya <: Complex ? complex.(randn(n,n), randn(n,n)) : randn(n,n)) |> z -> chol(z'z) |> z -> uplo == :U ? z : ctranspose(z)))
        b = convert(Matrix{eltyb}, eltya <: Complex ? real(A)*ones(n,2) : A*ones(n,2))
        SA = t(m(A.data))
        Sx = SA \ v(b[:, 1])
        x = A \ b[:, 1]
        @test Sx isa StaticVector # test not falling back to Base
        @test Sx ≈ x
        @test eltype(Sx) == eltype(x)
        SX = SA \ u(b)
        X = A \ b
        @test SX isa StaticMatrix # test not falling back to Base
        @test SX ≈ X
        @test eltype(SX) == eltype(X)
    end
end
