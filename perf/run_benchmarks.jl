benchmark_suite = true

global N = 0

for n = [4, 16]
    N = n
    include("benchmark2.jl")
end

benchmark_suite = false;
