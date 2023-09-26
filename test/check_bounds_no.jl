module TestCheckBoundsNo

using Test
using StaticArrays

# https://github.com/JuliaArrays/StaticArrays.jl/issues/1155
@testset "Issue #1155" begin
  u = @inferred(SVector(1, 2))
  v = @inferred(SVector(3.0, 4.0))
  a = 1.0
  b = 2
  result = @inferred(a * u + b * v)
  @test result ≈ @inferred(SVector(7, 10))
end

end # module
