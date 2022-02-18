module BenchMatrixOps

import Random
using BenchmarkTools
using LinearAlgebra
using StaticArrays

const suite = BenchmarkGroup()
const matrix_sizes = if haskey(ENV, "GITHUB_EVENT_PATH")
    (1, 2, 3, 4, 10)
else
    1:10
end

# Use same arrays across processes (at least with the same Julia version):
Random.seed!(1234)

# Unary operators
for f in [det, inv, exp]
    s1 = suite["$f"] = BenchmarkGroup()
    for N in matrix_sizes
        SA = @SMatrix rand(N, N)
        A = Array(SA)
        s2 = s1[string(N, pad=2)] = BenchmarkGroup()
        s2["SMatrix"] = @benchmarkable $f($SA)
        s2["Matrix"] = @benchmarkable $f($A)
    end
end

# Binary operators
for f in [*, \]
    s1 = suite["$f"] = BenchmarkGroup()
    for N in matrix_sizes
        SA = @SMatrix rand(N, N)
        SB = @SMatrix rand(N, N)
        A = Array(SA)
        B = Array(SB)
        s2 = s1[string(N, pad=2)] = BenchmarkGroup()
        s2["SMatrix"] = @benchmarkable $f($SA, $SB)
        s2["Matrix"] = @benchmarkable $f($A, $B)
    end
end

# Multiply-add
function benchmark_matmul(s,N1,N2,ArrayType)
    if ArrayType <: MArray
        Mat = MMatrix
        A = rand(Mat{N1,N2})
        B = rand(Mat{N2,N2})
        C = rand(Mat{N1,N2})
        label = "MMatrix"
    elseif ArrayType <: SizedArray
        Mat = SizedMatrix
        A = rand(Mat{N1,N2})
        B = rand(Mat{N2,N2})
        C = rand(Mat{N1,N2})
        label = "SizedMatrix"
    elseif ArrayType <: Array
        A = rand(N1,N2)
        B = rand(N2,N2)
        C = rand(N1,N2)
        label = "Matrix"
    end
    α,β = 1.0, 1.0
    s1 = s["mul!(C,A,B)"][string(N1, pad=2) * string(N2, pad=2)] = BenchmarkGroup()
    s2 = s["mul!(C,A,B,α,β)"][string(N1, pad=2) * string(N2, pad=2)] = BenchmarkGroup()
    s3 = s["mul!(B,A',C)"][string(N1, pad=2) * string(N2, pad=2)] = BenchmarkGroup()
    s4 = s["mul!(B,A',C,α,β)"][string(N1, pad=2) * string(N2, pad=2)] = BenchmarkGroup()

    s1[label] = @benchmarkable mul!($C,$A,$B)
    s2[label] = @benchmarkable mul!($C,$A,$B,$α,$β)
    s3[label] = @benchmarkable mul!($B,Transpose($A),$C)
    s4[label] = @benchmarkable mul!($B,Transpose($A),$C,$α,$β)
end

begin
    suite["mul!(C,A,B)"] = BenchmarkGroup(["inplace", "multiply-add"])
    suite["mul!(C,A,B,α,β)"] = BenchmarkGroup(["inplace", "multiply-add"])
    suite["mul!(B,A',C)"] = BenchmarkGroup(["inplace", "multiply-add"])
    suite["mul!(B,A',C,α,β)"] = BenchmarkGroup(["inplace", "multiply-add"])
    for N in matrix_sizes
        for Mat in (MMatrix, SizedMatrix, Matrix)
            benchmark_matmul(suite, N+1, N, Mat)
        end
    end
end


end  # module
BenchMatrixOps.suite
