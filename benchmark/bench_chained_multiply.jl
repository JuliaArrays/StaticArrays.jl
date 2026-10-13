# Benchmark of the compile-time cost model for chained `*` of static matrices
# against explicitly forcing left (and right) association.
#
# `A * B * C` and `A * B * C * D` dispatch to the generated `_mul_chain`, which
# picks the cheapest parenthesization from the static sizes.  `(A*B)*C` and
# `((A*B)*C)*D` force left association and therefore bypass the cost model.
#
# The file ends with a `BenchmarkGroup` so that `benchmarks.jl` can pick it up.
# Run `BenchmarkChainedMultiply.main()` (or `report()`) for a readable summary.

module BenchmarkChainedMultiply

using StaticArrays, BenchmarkTools, LinearAlgebra, Printf

randmat(n, m) = rand(SMatrix{n,m,Float64})

chain3(a, b, c) = a * b * c
left3(a, b, c) = (a * b) * c
right3(a, b, c) = a * (b * c)

chain4(a, b, c, d) = a * b * c * d
left4(a, b, c, d) = ((a * b) * c) * d
right4(a, b, c, d) = a * (b * (c * d))

# (label, shape A, shape B, shape C)
const CASES3 = [
    ("3x cost beats L (10x2)(2x10)(10x2)", (10,2), (2,10), (10,2)),
    ("3x cost beats L (8x2)(2x12)(12x3)",  (8,2),  (2,12), (12,3)),
    ("3x L is optimal (2x10)(10x2)(2x10)", (2,10), (10,2), (2,10)),
    ("3x L is optimal (12x3)(3x2)(2x11)",  (12,3), (3,2),  (2,11)),
    ("3x tie -> L     (4x4)(4x4)(4x4)",    (4,4),  (4,4),  (4,4)),
]

# (label, shape A, shape B, shape C, shape D)
const CASES4 = [
    ("4x picks (A(BC))D  (4x3)(3x5)(5x2)(2x6)",     (4,3),  (3,5),  (5,2),  (2,6)),
    ("4x picks A(B(CD))  (6x2)(2x3)(3x5)(5x2)",     (6,2),  (2,3),  (3,5),  (5,2)),
    ("4x picks (A(BC))D  (10x2)(2x10)(10x2)(2x10)", (10,2), (2,10), (10,2), (2,10)),
    ("4x all-tie -> L (4x4)^4",                      (4,4),  (4,4),  (4,4),  (4,4)),
]

function make_suite()
    suite = BenchmarkGroup()
    for (label, sA, sB, sC) in CASES3
        A, B, C = randmat(sA...), randmat(sB...), randmat(sC...)
        group = BenchmarkGroup()
        group["cost-based"]  = @benchmarkable chain3($A, $B, $C)
        group["forced-left"] = @benchmarkable left3($A, $B, $C)
        group["forced-right"] = @benchmarkable right3($A, $B, $C)
        suite[label] = group
    end
    for (label, sA, sB, sC, sD) in CASES4
        A, B, C, D = randmat(sA...), randmat(sB...), randmat(sC...), randmat(sD...)
        group = BenchmarkGroup()
        group["cost-based"]  = @benchmarkable chain4($A, $B, $C, $D)
        group["forced-left"] = @benchmarkable left4($A, $B, $C, $D)
        group["forced-right"] = @benchmarkable right4($A, $B, $C, $D)
        suite[label] = group
    end
    return suite
end

const suite = make_suite()

# The parenthesization the cost model selects, computed with the ported model.
function chosen3(sA, sB, sC)
    Sa, Sb, Sc = Size{sA}(), Size{sB}(), Size{sC}()
    return StaticArrays._mul_cost((Sa, Sb), Sc) <= StaticArrays._mul_cost(Sa, (Sb, Sc)) ?
        "((AB)C)" : "(A(BC))"
end

function chosen4(sA, sB, sC, sD)
    Sa, Sb, Sc, Sd = Size{sA}(), Size{sB}(), Size{sC}(), Size{sD}()
    costs = (StaticArrays._mul_cost((Sa, Sb), (Sc, Sd)),
             StaticArrays._mul_cost(((Sa, Sb), Sc), Sd),
             StaticArrays._mul_cost(Sa, (Sb, (Sc, Sd))),
             StaticArrays._mul_cost((Sa, (Sb, Sc)), Sd),
             StaticArrays._mul_cost(Sa, ((Sb, Sc), Sd)))
    names = ("(AB)(CD)", "((AB)C)D", "(A(B(CD)))", "((A(BC))D)", "(A((BC)D))")
    cmin = minimum(costs)
    for i in (2, 1, 3, 4, 5) # static tie-break: prefer left association
        costs[i] == cmin && return names[i]
    end
end

"""
    report(; params = nothing)

Run the suite and print the median time (ns) of the cost-based chain, the
explicitly left-associated chain, and the speed-up `left / cost`.
"""
function report(; params = nothing)
    params === nothing || loadparams!(suite, params, :evals, :samples)
    results = run(suite)
    @printf("%-44s %10s %10s %10s  %s\n",
            "case", "cost (ns)", "left (ns)", "right (ns)", "chosen")
    for (label, sA, sB, sC) in CASES3
        group = results[label]
        t_cost = median(group["cost-based"]).time
        t_left = median(group["forced-left"]).time
        t_right = median(group["forced-right"]).time
        @printf("%-44s %10d %10d %10d  left/cost = %4.2fx  %s\n",
                label, round(Int, t_cost), round(Int, t_left), round(Int, t_right),
                t_left / t_cost, chosen3(sA, sB, sC))
    end
    for (label, sA, sB, sC, sD) in CASES4
        group = results[label]
        t_cost = median(group["cost-based"]).time
        t_left = median(group["forced-left"]).time
        t_right = median(group["forced-right"]).time
        @printf("%-44s %10d %10d %10d  left/cost = %4.2fx  %s\n",
                label, round(Int, t_cost), round(Int, t_left), round(Int, t_right),
                t_left / t_cost, chosen4(sA, sB, sC, sD))
    end
    return results
end

main() = report()

end # module
BenchmarkChainedMultiply.suite
