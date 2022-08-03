@testset "Scalar" begin
    @test Scalar(2) .* [1, 2, 3] == [2, 4, 6]
    @test Scalar([1 2; 3 4]) .+ [[1 1; 1 1], [2 2; 2 2]] == [[2 3; 4 5], [3 4; 5 6]]
    @test (Scalar(1) + Scalar(1.0))::Scalar{Float64} ≈ Scalar(2.0)
    @test_throws ErrorException Scalar(2)[2]
    @test Scalar(2)[] == 2
    @testinf Tuple(Scalar(2)) === (2,)
    @testinf Tuple(convert(Scalar{Float64}, [2.0])) === (2.0,)
    a = Array{Float64, 0}(undef)
    a[] = 2
    @test Scalar(a)[] == 2
    s = Scalar(a)
    @test convert(typeof(s), s) === s
    @test Scalar(SVector(1,2,3))[] === SVector(1,2,3)
    @test Scalar(MArray{Tuple{}}(1)) === Scalar(1)
end

@testset "issue #809" begin
    @test_throws DimensionMismatch Scalar(1, 2)
end
