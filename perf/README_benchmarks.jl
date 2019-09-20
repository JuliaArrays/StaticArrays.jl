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
           ("Matrix multiplication              ", *,        (A, A),     (SA, SA)),
           ("Matrix multiplication (mutating)   ", mul!,     (B, A, A),  (MB, MA, MA)),
           ("Matrix addition                    ", +,        (A, A),     (SA, SA)),
           ("Matrix addition (mutating)         ", add!,     (B, A, A),  (MB, MA, MA)),
           ("Matrix determinant                 ", det,      (A,),       (SA,)),
           ("Matrix inverse                     ", inv,      (A,),       (SA,)),
           ("Matrix symmetric eigendecomposition", eigen,    (A,),       (SA,)),
           ("Matrix Cholesky decomposition      ", cholesky, (A,),       (SA,)),
           ("Matrix LU decomposition            ", lu,       (A,),       (SA,)),
           ("Matrix QR decomposition            ", qr,       (A,),       (SA,)),
    ]
    for (name, op, Aargs, SAargs) in ops
        # We load from Ref's here to avoid the compiler completely removing the
        # benchmark in some cases.
        #
        # Like any microbenchmark, the speedups you see here should only be
        # taken as roughly indicative of the speedup you may see in real code.
        if length(Aargs) == 1
            A1  = Ref(Aargs[1])
            SA1 = Ref(SAargs[1])
            speedup = @belapsed($op($A1[])) / @belapsed($op($SA1[]))
        elseif length(Aargs) == 2
            A1  = Ref(Aargs[1])
            A2  = Ref(Aargs[2])
            SA1 = Ref(SAargs[1])
            SA2 = Ref(SAargs[2])
            speedup = @belapsed($op($A1[], $A2[])) / @belapsed($op($SA1[], $SA2[]))
        elseif length(Aargs) == 3
            A1  = Ref(Aargs[1])
            A2  = Ref(Aargs[2])
            A3  = Ref(Aargs[3])
            SA1 = Ref(SAargs[1])
            SA2 = Ref(SAargs[2])
            SA3 = Ref(SAargs[3])
            speedup = @belapsed($op($A1[], $A2[], $A3[])) / @belapsed($op($SA1[], $SA2[], $SA3[]))
        else
        end
        println(name*" -> $(round(speedup, digits=1))x speedup")
    end
end

simple_bench(3)
