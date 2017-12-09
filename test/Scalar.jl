using StaticArrays, Base.Test

@testset "Scalar" begin
    @test Scalar(2) .* [1, 2, 3] == [2, 4, 6]
    @test Scalar([1 2; 3 4]) .+ [[1 1; 1 1], [2 2; 2 2]] == [[2 3; 4 5], [3 4; 5 6]]
    @test (Scalar(1) + Scalar(1.0))::Scalar{Float64} ≈ Scalar(2.0)
    @test_throws ErrorException Scalar(2)[2]
    @test Scalar(2)[] == 2
    @test Tuple(Scalar(2)) == (2,)
    @test Tuple(convert(Scalar{Float64}, [2.0])) == (2.0,)
    @compat a = Array{Float64, 0}(uninitialized)
    a[] = 2
    @test Scalar(a)[] == 2
end
