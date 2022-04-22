module BenchQR

using BenchmarkTools
using LinearAlgebra
using StaticArrays

const suite = BenchmarkGroup()

for K = [2, 3, 4, 8, 10, 12]
    a = rand(SMatrix{K,K,Float64,K*K})
    m = Matrix(a)
    s = suite["S=$K"] = BenchmarkGroup()
    s["SMatrix"] = @benchmarkable qr($a)
    s["Matrix"] = @benchmarkable qr($m)
end

end  # module
BenchQR.suite
