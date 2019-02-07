@testset "MScalar" begin
    @test MScalar(2) .* [1, 2, 3] == [2, 4, 6]
    @test_throws DimensionMismatch (MScalar(2) .= (Scalar(2) .* [1, 2, 3])) == [2, 4, 6]
    @test MScalar([1 2; 3 4]) .+ [[1 1; 1 1], [2 2; 2 2]] == [[2 3; 4 5], [3 4; 5 6]]
    @test (MScalar(1) + MScalar(1.0))::MScalar{Float64} ≈ MScalar(2.0)
    @test_throws BoundsError MScalar(2)[2]
    @test MScalar(2)[] == 2
    x = MScalar(2)
    x[] = 1
    @test x[] == 1
    @testinf Tuple(MScalar(2)) === (2,)
    @testinf Tuple(convert(MScalar{Float64}, [2.0])) === (2.0,)
    a = Array{Float64, 0}(undef)
    a[] = 2
    @test MScalar(a)[] == 2
    s = MScalar(a)
    @test convert(typeof(s), s) === s
    s[] = 1
    @test s[] == 1
    s .= 42
    @test s[] == 42
end
