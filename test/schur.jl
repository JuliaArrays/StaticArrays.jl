using StaticArrays, Test, LinearAlgebra

@testset "Schur decomposition" begin
    matrices = (
        (@SMatrix [1.]),
        (@SMatrix [1. 2.; 3. 4.]),
        (@SMatrix [1. 2. 3.; -1. 2. 3.; 0. 0. 1.]),

        (@MMatrix [1.]),
        (@MMatrix [1. 2.; 3. 4.]),
        (@MMatrix [1. 2. 3.; -1. 2. 3.; 0. 0. 1.])
    )
    for m in matrices
        s1 = schur(Array(m))
        s2 = schur(m)
        @test s1.T ≈ Array(s2.T)
        @test s1.Z ≈ Array(s2.Z)
        @test s1.values ≈ Array(s2.values)
    end
end
