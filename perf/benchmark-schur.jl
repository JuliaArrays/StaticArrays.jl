using StaticArrays
using BenchmarkTools

a = m = 0

for K = 1:22
    a = rand(MMatrix{K,K,Float64,K*K})
    m = Matrix(a)
    print("Size($K,$K)\n  Compilation time:")
    @time schur(a)
    @btime schur($a)
    @btime schur($m)
end
