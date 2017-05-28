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
end

@testset "Solving triangular system" begin
    for n in (1,2,3,4),
          (t1, uplo1) in ((UpperTriangular, :U),
                          (LowerTriangular, :L)),
            (m, v, u) in ((SMatrix{n, n}, SVector{n}, SMatrix{n, 2}), (MMatrix{n,n}, MVector{n}, SMatrix{n, 2})),
                elty in (Float32, Float64, Int)

        eval(quote
            A = $(t1)($elty == Int ? rand(1:7, $n, $n) : convert(Matrix{$elty}, randn($n, $n)) |> t -> chol(t't) |> t -> $(uplo1 == :U) ? t : ctranspose(t))
            b = convert(Matrix{$elty}, A*ones($n, 2))
            SA = $t1($m(A.data))
            @test SA \ $v(b[:, 1]) ≈ A \ b[:, 1]
            @test SA \ $u(b) ≈ A \ b
        end)
    end
end
