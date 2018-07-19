using BenchmarkTools
using LinearAlgebra
using StaticArrays

add!(C, A, B) = (C .= A .+ B)

function simple_bench(N, T=Float64)
    A = rand(T,N,N)
    A = A'*A
    B = copy(A)
    SA = SMatrix{N,N}(A)
    MA = MMatrix{N,N}(A)
    MB = copy(MA)

    print("""
============================================
    Benchmarks for $N×$N $T matrices
============================================
""")
    ops = [
        ("Matrix multiplication              ", *, (A, A), (SA, SA)),
        ("Matrix multiplication (mutating)   ", mul!, (B, A, A), (MB, MA, MA)),
        ("Matrix addition                    ", +, (A, A), (SA, SA)),
        ("Matrix addition (mutating)         ", add!, (B, A, A), (MB, MA, MA)),
        ("Matrix determinant                 ", det, A, SA),
        ("Matrix inverse                     ", inv, A, SA),
        ("Matrix symmetric eigendecomposition", eigen, A, SA),
        ("Matrix Cholesky decomposition      ", cholesky, A, SA)
    ]
    for (name, op, Aargs, SAargs) in ops
        if Aargs isa Tuple && length(Aargs) == 2
            speedup = @belapsed($op($Aargs[1], $Aargs[2])) / @belapsed($op($SAargs[1], $SAargs[2]))
        elseif Aargs isa Tuple && length(Aargs) == 3
            speedup = @belapsed($op($Aargs[1], $Aargs[2], $Aargs[3])) / @belapsed($op($SAargs[1], $SAargs[2], $SAargs[3]))
        else
            speedup = @belapsed($op($Aargs)) / @belapsed($op($SAargs))
        end
        println(name*" -> $(round(speedup, digits=1))x speedup")
    end
end

simple_bench(3)
