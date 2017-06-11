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
