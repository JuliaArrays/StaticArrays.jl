module BenchSort

using BenchmarkTools
using Random: rand!
using StaticArrays
using StaticArrays: BitonicSort

const SUITE = BenchmarkGroup()

# 1 second is sufficient for reasonably consistent timings.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

const LEN = 1000

const Floats = (Float16, Float32, Float64)
const Ints = (Int8, Int16, Int32, Int64, Int128)
const UInts = (UInt8, UInt16, UInt32, UInt64, UInt128)

map_sort!(vs; kwargs...) = map!(v -> sort(v; kwargs...), vs, vs)

addgroup!(SUITE, "BitonicSort")

g = addgroup!(SUITE["BitonicSort"], "SVector")
for lt in (isless, <)
    n = 1
    while (n = nextprod([2, 3], n + 1)) <= 24
        for T in (Floats..., Ints..., UInts...)
            (lt === <) && (T <: Integer) && continue  # For Integers, isless is <.
            vs = Vector{SVector{n, T}}(undef, LEN)
            g[lt, n, T] = @benchmarkable(
                map_sort!($vs; alg=BitonicSort, lt=$lt),
                evals=1,  # Redundant on @benchmarkable as of BenchmarkTools 1.1.3.
                # We need evals=1 so that setup runs before every eval. But PkgBenchmark
                # always `tunes!` benchmarks before running, which overrides this. As a
                # workaround, use the unhygienic symbol `__params` to set evals just before
                # execution at
                # https://github.com/JuliaCI/BenchmarkTools.jl/blob/v1.1.3/src/execution.jl#L482
                # See also: https://github.com/JuliaCI/PkgBenchmark.jl/issues/120
                setup=(__params.evals = 1; rand!($vs)),
            )
        end
    end
end

g = addgroup!(SUITE["BitonicSort"], "MVector")
for (lt, n, T) in ((isless, 16, Int64), (isless, 16, Float64), (<, 16, Float64))
    vs = Vector{MVector{n, T}}(undef, LEN)
    g[lt, n, T] = @benchmarkable(
        map_sort!($vs; alg=BitonicSort, lt=$lt),
        evals=1,
        setup=(__params.evals = 1; rand!($vs)),
    )
end

g = addgroup!(SUITE["BitonicSort"], "SizedVector")
for (lt, n, T) in ((isless, 16, Int64), (isless, 16, Float64), (<, 16, Float64))
    vs = Vector{SizedVector{n, T, Vector{T}}}(undef, LEN)
    g[lt, n, T] = @benchmarkable(
        map_sort!($vs; alg=BitonicSort, lt=$lt),
        evals=1,
        setup=(__params.evals = 1; rand!($vs)),
    )
end

# @generated to unroll the tuple.
@generated function floats_nans(::Type{SVector{N, T}}, p) where {N, T}
    exprs = (:(ifelse(rand(Float32) < p, T(NaN), rand(T))) for _ in 1:N)
    return quote
        Base.@_inline_meta
        return SVector(($(exprs...),))
    end
end

function floats_nans!(vs::Vector{SVector{N, T}}, p) where {N, T}
    for i in eachindex(vs)
        @inbounds vs[i] = floats_nans(SVector{N, T}, p)
    end
    return vs
end

g = addgroup!(SUITE["BitonicSort"], "NaNs")
for p in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
    (lt, n, T) = (isless, 16, Float64)
    vs = Vector{SVector{n, T}}(undef, LEN)
    g[lt, n, T, p] = @benchmarkable(
        map_sort!($vs; alg=BitonicSort, lt=$lt),
        evals=1,
        setup=(__params.evals = 1; floats_nans!($vs, $p)),
    )
end

end # module BenchSort

# Allow PkgBenchmark.benchmarkpkg to call this file directly.
@isdefined(SUITE) || (SUITE = BenchSort.SUITE)

BenchSort.SUITE
