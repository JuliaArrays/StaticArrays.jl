module BenchMatrixOps

using BenchmarkTools
using LinearAlgebra
using Random
using StaticArrays

const suite = BenchmarkGroup()
const Nmax = 20

# Use same arrays across processes (at least with the same Julia version):
const RNG = MersenneTwister(1234)

# Unary operators
for f in [det, inv, exp]
    s1 = suite["$f"] = BenchmarkGroup()
    for N in 1:Nmax
        SA = @SMatrix rand(RNG, N, N)
        A = Array(SA)
        s2 = s1["$N"] = BenchmarkGroup()
        s2["SMatrix"] = @benchmarkable $f($SA)
        s2["Matrix"] = @benchmarkable $f($A)
    end
end

# Binary operators
for f in [*, \]
    s1 = suite["$f"] = BenchmarkGroup()
    for N in 1:Nmax
        SA = @SMatrix rand(RNG, N, N)
        SB = @SMatrix rand(RNG, N, N)
        A = Array(SA)
        B = Array(SB)
        s2 = s1["$N"] = BenchmarkGroup()
        s2["SMatrix"] = @benchmarkable $f($SA, $SB)
        s2["Matrix"] = @benchmarkable $f($A, $B)
    end
end

end  # module
BenchMatrixOps.suite
