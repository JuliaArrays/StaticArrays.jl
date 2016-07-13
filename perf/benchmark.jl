fsa = true

using StaticArrays
@static if fsa
    using FixedSizeArrays
end

if !isdefined(:benchmark_suite) || benchmark_suite == false
    N = 4
end

M_f = div(10^9, N^3)
M_g = div(2*10^8, N^2)

# Size


A = rand(N,N)
B = rand(N,N)
As = SMatrix{N,N}(A)
Bs = SMatrix{N,N}(B)
Am = MMatrix{N,N}(A)
Bm = MMatrix{N,N}(B)
@static if fsa
    Af = Mat(ntuple(j -> ntuple(i->A[i,j], N), N)) # there is a bug in FixedSizeArrays Mat constructor (13 July 2016)
    Bf = Mat(ntuple(j -> ntuple(i->B[i,j], N), N))
end

if !isdefined(:f_mut)
    f(n::Integer, A, B) = @inbounds (C = A*B; for i = 1:n; C += A*B; end; return C)
    f_mut(n::Integer, A, B) = @inbounds (C = A*B; tmp = similar(C); for i = 1:n; A_mul_B!(tmp, A, B); map!(+, C, C, tmp); end; return C)
    g(n::Integer, A, B) = @inbounds (C = A + B; for i = 1:n; C = C + B; end; return C)
    g_mut(n::Integer, A, B) = @inbounds (C = A + B; for i = 1:n; @inbounds broadcast!(+, C, C, B); end; return C)
end

# Don't count these compilations either
C = f(2, A, B)
C_mut = f_mut(2, A, B)
# @assert C_mut ≈ C

C = g(2, A, B)
C_mut = g_mut(2, A, B)
# @assert C_mut ≈ C

println("=====================================")
println("    Benchmarks for $N×$N matrices")
println("=====================================")
print("StaticArrays compilation time (×3):")
@time eval(quote
    # Warmup and some checks
    C = f(2, A, B)
    Cs = f(2, As, Bs)
    Cm = f(2, Am, Bm)
    Cm_mut = f_mut(2, Am, Bm)

    Cs::SMatrix
    #@assert Cs ≈ C
    Cm::MMatrix
    #@assert Cm ≈ C

    C = g(2, A, B)
    Cs = g(2, As, Bs)
    Cm = g(2, Am, Bm)
    Cm_mut = g_mut(2, Am, Bm)

    Cs::SMatrix
    #@assert Cs == C
    Cm::MMatrix
    #@assert Cm == C
    Cm_mut::MMatrix
    #@assert Cm_mut == C
end)
@static if fsa
    print("FixedSizeArrays compilation time:  ")
    @time eval(quote
        C = f(2, A, B)
        Cf = f(2, Af, Bf)
        Cf::Mat
        #@assert Cf == C

        C = g(2, A, B)
        Cf = g(2, Af, Bf)
        Cf::Mat
        #@assert Cf == C
    end)
end
println()

# Do the performance tests

println("Matrix multiplication and accumulation")
println("--------------------------------------")
begin
   print("Array             ->"); @time f(M_f, A, B)
   print("Array (mutating)  ->"); @time f_mut(M_f, A, B)
   print("SArray            ->"); @time f(M_f, As, Bs)
   print("MArray            ->"); @time f(M_f, Am, Bm)
   print("MArray (mutating) ->"); @time f_mut(M_f, Am, Bm)
   @static if fsa
       print("Mat               ->"); @time f(M_f, Af, Bf)
   end
end
println()

println("Matrix addition and accumulation")
println("--------------------------------")
begin
   print("Array             ->"); @time g(M_g, A, B)
   print("Array (mutating)  ->"); @time g_mut(M_g, A, B) # broadcast! seems to be broken!
   print("SArray            ->"); @time g(M_g, As, Bs)
   print("MArray            ->"); @time g(M_g, Am, Bm)
   print("MArray (mutating) ->"); @time g_mut(M_g, Am, Bm)
   @static if fsa
       print("Mat               ->"); @time g(M_g, Af, Bf)
   end
end
println()
