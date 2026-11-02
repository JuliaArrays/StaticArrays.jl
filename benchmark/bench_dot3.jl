module BenchDot3

# Three-argument `dot(x, A, y)` compared to the generic method from LinearAlgebra
# (the implementation used before StaticArrays had a specialized method) and to
# `dot(x, A * y)`. Run this file directly to print a table that flags sizes where the
# specialized method is slower than either alternative:
#
#     julia --project=benchmark benchmark/bench_dot3.jl

import Random
using BenchmarkTools
using ForwardDiff: Dual
using LinearAlgebra
using Printf
using StaticArrays

const suite = BenchmarkGroup()

# Besides square matrices, this includes the shapes next to the size limits of `_dot3`.
const shapes = if haskey(ENV, "GITHUB_EVENT_PATH")
    [(2, 2), (3, 3), (4, 4), (8, 8), (15, 15)]
else
    [[(n, n) for n in 1:15]; [(3, 8), (8, 3), (4, 12), (15, 2), (2, 15)];
     [(5, 5), (6, 5), (6, 6), (11, 10), (12, 9), (12, 10), (16, 11), (18, 11), (29, 2), (30, 2), (64, 2), (256, 1)]] |> unique
end

const eltypes = (Float64, Float32, ComplexF64, ComplexF32,
                 Dual{Nothing,Float64,1}, Dual{Nothing,Float64,3}, Dual{Nothing,Float64,7})

rand_element(::Type{T}) where {T} = rand(T)
rand_element(::Type{Dual{Nothing,V,K}}) where {V,K} = Dual{Nothing}(rand(V), ntuple(_ -> rand(V), K)...)

generic_dot(x, A, y) = invoke(dot, Tuple{AbstractVector, AbstractMatrix, AbstractVector}, x, A, y)
matvec_dot(x, A, y) = dot(x, A * y)

const implementations = (("dot(x,A,y)", dot), ("generic", generic_dot), ("dot(x,A*y)", matvec_dot))

Random.seed!(1234)

for T in eltypes
    s1 = suite[string(T)] = BenchmarkGroup()
    for (M, N) in shapes
        x = SVector{M,T}(ntuple(_ -> rand_element(T), M))
        A = SMatrix{M,N,T}(ntuple(_ -> rand_element(T), M * N))
        y = SVector{N,T}(ntuple(_ -> rand_element(T), N))
        s2 =s1[string(M, pad=2) * "x" * string(N, pad=2)] = BenchmarkGroup()
        for (label, f) in implementations
            s2[label] = @benchmarkable $f($x, $A, $y)
        end
    end
end

# Print minimum times and mark sizes where `dot(x,A,y)` is slower than an alternative by
# more than the relative tolerance `rtol` plus an absolute tolerance `atol` (in ns).
function compare(; rtol = 0.05, atol = 0.3, seconds = 0.5)
    tune!(suite)
    results = run(suite; seconds = seconds)
    nregressions = 0
    for T in eltypes
        println(T)
        r1 = results[string(T)]
        for key in sort(collect(keys(r1)))
            times = [time(minimum(r1[key][label])) for (label, _) in implementations]
            slower = times[1] > (1 + rtol) * minimum(times[2:end]) + atol
            nregressions += slower
            @printf("  %s  dot(x,A,y) %8.2f ns  generic %8.2f ns  dot(x,A*y) %8.2f ns%s\n",
                key, times..., slower ? "  <-- regression" : "")
        end
    end
    println(nregressions == 0 ? "No regressions." : "$nregressions regression(s).")
    return nregressions
end

end  # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(BenchDot3.compare() == 0 ? 0 : 1)
end

BenchDot3.suite
