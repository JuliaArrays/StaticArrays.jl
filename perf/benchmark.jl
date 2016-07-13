using StaticArrays
using FixedSizeArrays

N = 8

A = rand(N,N)
B = rand(N,N)
As = SMatrix{N,N}(A)
Bs = SMatrix{N,N}(B)
Am = MMatrix{N,N}(A)
Bm = MMatrix{N,N}(B)
Af = Mat(ntuple(j -> ntuple(i->A[i,j], N), N)) # there is a bug in FixedSizeArrays (13 July 2016)
Bf = Mat(ntuple(j -> ntuple(i->B[i,j], N), N))

f(n::Integer, A, B) = @inbounds (C = A*B; for i = 1:n; C += A*B; end; return C)
f_mut(n::Integer, A, B) = @inbounds (C = A*B; tmp = similar(C); for i = 1:n; A_mul_B!(tmp, A, B); map!(+, C, C, tmp); end; return C)
g(n::Integer, A, B) = @inbounds (C = A + B; for i = 1:n; C = C + B; end; return C)
g_mut(n::Integer, A, B) = @inbounds (C = A + B; for i = 1:n; @inbounds broadcast!(+, C, C, B); end; return C)



# Warmup and some checks
C = f(10, A, B)
C_mut = f_mut(10, A, B)
Cs = f(10, As, Bs)
Cm = f(10, Am, Bm)
Cm_mut = f_mut(10, Am, Bm)
Cf = f(10, Af, Bf)

@assert C_mut == C
Cs::SMatrix
@assert Cs == C
Cm::MMatrix
@assert Cm == C
Cf::Mat
@assert Cf == C

C = g(10, A, B)
C_mut = g_mut(10, A, B)
Cs = g(10, As, Bs)
Cm = g(10, Am, Bm)
Cm_mut = g_mut(10, Am, Bm)
Cf = g(10, Af, Bf)

@assert C_mut == C
Cs::SMatrix
@assert Cs == C
Cm::MMatrix
@assert Cm == C
Cm_mut::MMatrix
@assert Cm_mut == C
Cf::Mat
@assert Cf == C

# Do the performance tests

println("Matrix multiplication and acumulation")
println("-------------------------------------")
begin
   print("Array             -> "); @time f(10^6, A, B)
   print("Array (mutating)  -> "); @time f_mut(10^6, A, B)
   print("SArray            -> "); @time f(10^6, As, Bs)
   print("MArray            -> "); @time f(10^6, Am, Bm)
   print("MArray (mutating) -> "); @time f_mut(10^6, Am, Bm)
   print("Mat               -> 10 ×"); @time f(10^5, Af, Bf)
end
println()

println("Matrix addition and acumulation")
println("-------------------------------")
begin
   print("Array             -> "); @time g(10^6, A, B)
   print("Array (mutating)  -> 10 ×"); @time g_mut(10^6, A, B)
   print("SArray            -> "); @time g(10^6, As, Bs)
   print("MArray            -> "); @time g(10^6, Am, Bm)
   print("MArray (mutating) -> "); @time g_mut(10^6, Am, Bm)
   print("Mat               -> "); @time g(10^6, Af, Bf)
end
println()
