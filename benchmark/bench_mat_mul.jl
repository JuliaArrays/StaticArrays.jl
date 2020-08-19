using StaticArrays
using BenchmarkTools
using LinearAlgebra
using Printf

suite = BenchmarkGroup()

mul_wrappers = [
    (m -> m, "ident  "),
    (m -> Symmetric(m, :U), "sym-u  "),
    (m -> Hermitian(m, :U), "herm-u "),
    (m -> UpperTriangular(m), "up-tri "),
    (m -> LowerTriangular(m), "lo-tri "),
    (m -> UnitUpperTriangular(m), "uup-tri"),
    (m -> UnitLowerTriangular(m), "ulo-tri"),
    (m -> Adjoint(m), "adjoint"),
    (m -> Transpose(m), "transpo")]

for N in [2, 4, 8, 10, 16]

    matvecstr = @sprintf("mat-vec  %2d", N)
    matmatstr = @sprintf("mat-mat  %2d", N)
    matvec_mut_str = @sprintf("mat-vec! %2d", N)
    matmat_mut_str = @sprintf("mat-mat! %2d", N)

    suite[matvecstr] = BenchmarkGroup()
    suite[matmatstr] = BenchmarkGroup()
    suite[matvec_mut_str] = BenchmarkGroup()
    suite[matmat_mut_str] = BenchmarkGroup()


    A = randn(SMatrix{N,N,Float64})
    B = randn(SMatrix{N,N,Float64})
    bv = randn(SVector{N,Float64})
    for (wrapper_a, wrapper_name) in mul_wrappers
        thrown = false
        try
            wrapper_a(A) * bv
        catch e
            thrown = true
        end
        if !thrown
            suite[matvecstr][wrapper_name] = @benchmarkable $(wrapper_a(A)) * $bv
        end
    end

    for (wrapper_a, wrapper_a_name) in mul_wrappers, (wrapper_b, wrapper_b_name) in mul_wrappers
        thrown = false
        try
            wrapper_a(A) * wrapper_b(B)
        catch e
            thrown = true
        end
        if !thrown
            suite[matmatstr][wrapper_a_name * " * " * wrapper_b_name] = @benchmarkable $(wrapper_a(A)) * $(wrapper_b(B))
        end
    end

    C = randn(MMatrix{N,N,Float64})
    cv = randn(MVector{N,Float64})

    for (wrapper_a, wrapper_name) in mul_wrappers
        thrown = false
        try
            mul!(cv, wrapper_a(A), bv)
        catch e
            thrown = true
        end
        if !thrown
            suite[matvec_mut_str][wrapper_name] = @benchmarkable mul!($cv, $(wrapper_a(A)), $bv)
        end
    end

    for (wrapper_a, wrapper_a_name) in mul_wrappers, (wrapper_b, wrapper_b_name) in mul_wrappers
        thrown = false
        try
            wrapper_a(A) * wrapper_b(B)
        catch e
            thrown = true
        end
        if !thrown
            suite[matmat_mut_str][wrapper_a_name * " * " * wrapper_b_name] = @benchmarkable mul!($C, $(wrapper_a(A)), $(wrapper_b(B)))
        end
    end
end

function run_and_save(fname, make_params = true)
    if make_params
        tune!(suite)
        BenchmarkTools.save("params.json", params(suite))
    else
        loadparams!(suite, BenchmarkTools.load("params.json")[1], :evals, :samples)
    end
    results = run(suite, verbose = true)
    BenchmarkTools.save(fname, results)
end

function judge_results(m1, m2)
    results = Any[]
    for key1 in keys(m1)
        if !haskey(m2, key1)
            continue
        end
        for key2 in keys(m1[key1])
            if !haskey(m2[key1], key2) 
                continue
            end
            push!(results, (key1, key2, judge(median(m1[key1][key2]), median(m2[key1][key2]))))
        end
    end
    return results
end
