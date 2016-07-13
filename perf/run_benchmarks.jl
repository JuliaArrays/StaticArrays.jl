benchmark_suite = true

global N = 0

for n = 2:16
    N = n
    include("benchmark.jl")
end

benchmark_suite = false
