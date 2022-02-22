module BenchmarkMatMul

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
    (m -> Transpose(m), "transpo"),
    (m -> Diagonal(m), "diag   ")]

mul_wrappers_reduced = [
    (m -> m, "ident  "),
    (m -> Symmetric(m, :U), "sym-u  "),
    (m -> UpperTriangular(m), "up-tri "),
    (m -> Transpose(m), "transpo"),
    (m -> Diagonal(m), "diag   ")]

for N in [2, 4, 8, 10, 12]

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
    for (wrapper_a, wrapper_name) in mul_wrappers_reduced
        thrown = false
        try
            wrapper_a(A) * bv
        catch e
            thrown = true
        end
        if !thrown
            suite[matvecstr][wrapper_name] = @benchmarkable $(Ref(wrapper_a(A)))[] * $(Ref(bv))[]
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
            suite[matmatstr][wrapper_a_name * " * " * wrapper_b_name] = @benchmarkable $(Ref(wrapper_a(A)))[] * $(Ref(wrapper_b(B)))[]
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
            suite[matvec_mut_str][wrapper_name] = @benchmarkable mul!($cv, $(Ref(wrapper_a(A)))[], $(Ref(bv))[])
        end
    end

    for (wrapper_a, wrapper_a_name) in mul_wrappers, (wrapper_b, wrapper_b_name) in mul_wrappers
        thrown = false
        try
            mul!(C, wrapper_a(A), wrapper_b(B))
        catch e
            thrown = true
        end
        if !thrown
            suite[matmat_mut_str][wrapper_a_name * " * " * wrapper_b_name] = @benchmarkable mul!($C, $(Ref(wrapper_a(A)))[], $(Ref(wrapper_b(B)))[])
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

function full_benchmark(mul_wrappers, size_iter = 1:4, T = Float64)
    suite_full = BenchmarkGroup()
    for N in size_iter
        for M in size_iter
            a = randn(SMatrix{N,M,T})
            wrappers_a = N == M ? mul_wrappers : [mul_wrappers[1]]
            sa = Size(a)
            for K in size_iter
                b = randn(SMatrix{M,K,T})
                wrappers_b = M == K ? mul_wrappers : [mul_wrappers[1]]
                sb = Size(b)
                for (w_a, w_a_name) in wrappers_a
                    for (w_b, w_b_name) in wrappers_b
                        cur_str = @sprintf("mat-mat %s %s generic  (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                        suite_full[cur_str] = @benchmarkable StaticArrays.mul_generic($sa, $sb, $(Ref(w_a(a)))[], $(Ref(w_b(b)))[])
                        cur_str = @sprintf("mat-mat %s %s default  (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                        suite_full[cur_str] = @benchmarkable StaticArrays._mul($sa, $sb, $(Ref(w_a(a)))[], $(Ref(w_b(b)))[])
                        cur_str = @sprintf("mat-mat %s %s unrolled (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                        suite_full[cur_str] = @benchmarkable StaticArrays.mul_unrolled($sa, $sb, $(Ref(w_a(a)))[], $(Ref(w_b(b)))[])
                        if w_a_name != "diag   " && w_b_name != "diag   "
                            cur_str = @sprintf("mat-mat %s %s chunks   (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                            suite_full[cur_str] = @benchmarkable StaticArrays.mul_unrolled_chunks($sa, $sb, $(Ref(w_a(a)))[], $(Ref(w_b(b)))[])
                        end
                        if w_a_name == "ident  " && w_b_name == "ident  "
                            cur_str = @sprintf("mat-mat %s %s loop     (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                            suite_full[cur_str] = @benchmarkable StaticArrays.mul_loop($sa, $sb, $(Ref(w_a(a)))[], $(Ref(w_b(b)))[])
                        end
                    end
                end
            end
        end
    end
    results = run(suite_full, verbose = true)
    results_median = map(collect(results)) do res
        return (res[1], median(res[2]).time)
    end
    return results_median
end

function judge_this(new_time, old_time, tol, w_a_name, w_b_name, N, M, K, which)
    if new_time*tol < old_time
        msg = @sprintf("better for %s %s (%2d, %2d) x (%2d, %2d): %s", w_a_name, w_b_name, N, M, M, K, which)
        println(msg)
        println(">> ", new_time, " | ", old_time)
    end
end

function pick_best(results, mul_wrappers, size_iter; tol = 1.2)
    for N in size_iter
        for M in size_iter
            wrappers_a = N == M ? mul_wrappers : [mul_wrappers[1]]
            for K in size_iter
                wrappers_b = M == K ? mul_wrappers : [mul_wrappers[1]]
                for (w_a, w_a_name) in wrappers_a
                    for (w_b, w_b_name) in wrappers_b
                        cur_default = @sprintf("mat-mat %s %s default  (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                        default_time = results[cur_default]

                        cur_generic = @sprintf("mat-mat %s %s generic  (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                        generic_time = results[cur_generic]
                        judge_this(generic_time, default_time, tol, w_a_name, w_b_name, N, M, K, "generic")

                        cur_unrolled = @sprintf("mat-mat %s %s unrolled (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                        unrolled_time = results[cur_unrolled]
                        judge_this(unrolled_time, default_time, tol, w_a_name, w_b_name, N, M, K, "unrolled")
                        
                        if w_a_name != "diag   " && w_b_name != "diag   "
                            cur_chunks = @sprintf("mat-mat %s %s chunks   (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                            chunk_time = results[cur_chunks]
                            judge_this(chunk_time, default_time, tol, w_a_name, w_b_name, N, M, K, "chunks")
                        end
                        if w_a_name == "ident  " && w_b_name == "ident  "
                            cur_loop = @sprintf("mat-mat %s %s loop     (%2d, %2d) x (%2d, %2d)", w_a_name, w_b_name, N, M, M, K)
                            loop_time = results[cur_loop]
                            judge_this(loop_time, default_time, tol, w_a_name, w_b_name, N, M, K, "loop")
                        end
                    end
                end
            end
        end
    end
end

function run_1()
    return full_benchmark(mul_wrappers_reduced, [2, 3, 4, 5, 8, 9, 12])
end

end #module
BenchmarkMatMul.suite
