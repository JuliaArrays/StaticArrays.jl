benchmark_suite = true

global N = 0

for n = 2:16
    N = n
    include("benchmark2.jl")
end

benchmark_suite = false;
