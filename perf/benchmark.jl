using StaticArrays
using FixedSizeArrays

A = rand(4,4)
B = rand(4,4)
As = SMatrix{4,4}(A)
Bs = SMatrix{4,4}(B)
Am = MMatrix{4,4}(A)
Bm = MMatrix{4,4}(B)
Af = Mat{4,4}((1.,2.,3.,4.)) # there is a bug in FixedSizeArrays (13 July 2016)
Bf = Mat{4,4}((4.,3.,2.,1.))

f(n::Integer, A, B) = @inbounds (C = A*B; for i = 1:n; C += A*B; end; return C)
g(n::Integer, A, B) = @inbounds (C = A + B; for i = 1:n; C = C + A + B; end; return C)

# Warmup and some checks
C = f(10, A, B)
Cs = f(10, As, Bs)
Cm = f(10, Am, Bm)
Cf = f(10, Af, Bf)

Cs::SMatrix
@assert Cs == C
Cm::MMatrix
@assert Cm == C
Cf::Mat

C = g(10, A, B)
Cs = g(10, As, Bs)
Cm = g(10, Am, Bm)
Cf = g(10, Af, Bf)

Cs::SMatrix
@assert Cs == C
Cm::MMatrix
@assert Cm == C
Cf::Mat

# Do the performance tests

println("Matrix multiplication and acumulation")
println("-------------------------------------")
begin
   print("Array  -> "); @time f(10^7, A, B)
   print("SArray -> "); @time f(10^7, As, Bs)
   print("MArray -> "); @time f(10^7, Am, Bm)
   print("Mat    -> "); @time f(10^7, Af, Bf)
end
println()

println("Matrix addition and acumulation")
println("-------------------------------")
begin
   print("Array  -> "); @time g(10^7, A, B)
   print("SArray -> "); @time g(10^7, As, Bs)
   print("MArray -> "); @time g(10^7, Am, Bm)
   print("Mat    -> "); @time g(10^7, Af, Bf)
end
println()
