@testset "Solving linear system" begin

    @testset "Problem size: $n x $n. Matrix type: $m. Element type: $elty" for n in (1,2,3,4),
            (m, v) in ((SMatrix{n,n}, SVector{n}), (MMatrix{n,n}, MVector{n})),
                elty in (Float64, Int)

        A = elty.(rand(-99:2:99,n,n))
        b = A*ones(elty,n)
        @test m(A)\v(b) ≈ A\b
    end
end
