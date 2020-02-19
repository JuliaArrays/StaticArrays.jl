# Each file of the form "bench_$(name).jl" in this directory is `include`d and
# its last statement is assumed to be a `BenchmarkGroup`.  This group is added
# to the top-level group `SUITE` with the `$name` extracted from the file name.

using BenchmarkTools
const SUITE = BenchmarkGroup()
for file in sort(readdir(@__DIR__))
    if startswith(file, "bench_") && endswith(file, ".jl")
        SUITE[chop(file, head = length("bench_"), tail = length(".jl"))] =
            include(file)
    end
end
