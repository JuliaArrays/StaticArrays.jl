benchmark_suite = true
using LinearAlgebra

N = 0

for n = 2:16
    global N
    N = n
    include("benchmark2.jl")
end

benchmark_suite = false;
