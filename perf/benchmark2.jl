fsa = true

using StaticArrays
@static if fsa
    using FixedSizeArrays
end

if !isdefined(:N)
    N = 4
end

M_f = div(10^9, N^3)
M_g = div(2*10^8, N^2)

# Size

A = rand(N,N)
As = SMatrix{N,N}(A)
Am = MMatrix{N,N}(A)
@static if fsa
    Af = Mat(ntuple(j -> ntuple(i->A[i,j], N), N)) # there is a bug in FixedSizeArrays Mat constructor (13 July 2016)
end


if !isdefined(:f_mut_marray) || !isdefined(:benchmark_suite) || benchmark_suite == false
    f(n::Integer, A) = @inbounds (C = A; for i = 1:n; C = C*A; end; return C)
    f_unrolled{M}(n::Integer, A::Union{SMatrix{M,M},MMatrix{M,M}}) = @inbounds (C = A; for i = 1:n; C = StaticArrays.A_mul_B_unrolled(C,A); end; return C)
    f_unrolled_chunks{M}(n::Integer, A::Union{SMatrix{M,M},MMatrix{M,M}}) = @inbounds (C = A; for i = 1:n; C = StaticArrays.A_mul_B_unrolled_chunks(C,A); end; return C)
    f_via_sarray{M}(n::Integer, A::MMatrix{M,M})= @inbounds (C = A; for i = 1:n; C = MMatrix{M,M}(SMatrix{M,M}(C)*SMatrix{M,M}(A)); end; return C)
    f_mut_array(n::Integer, A) = @inbounds (C = copy(A); tmp = similar(A); for i = 1:n;  A_mul_B!(tmp, C, A); map!(identity, C, tmp); end; return C)
    f_mut_marray(n::Integer, A) = @inbounds (C = similar(A); C[:] = A[:]; tmp = similar(A); for i = 1:n; StaticArrays.A_mul_B_unrolled!(tmp, C, A); C.data = tmp.data; end; return C)
    f_blas_marray(n::Integer, A) = @inbounds (C = similar(A); C[:] = A[:]; tmp = similar(A); for i = 1:n; StaticArrays.A_mul_B_blas!(tmp, C, A); C.data = tmp.data; end; return C)

    g(n::Integer, A) = @inbounds (C = A; for i = 1:n; C = C + A; end; return C)
    g_mut(n::Integer, A) = @inbounds (C = similar(A); C[:] = A[:]; for i = 1:n; @inbounds map!(+, C, C, A); end; return C)
    g_via_sarray{M}(n::Integer, A::MMatrix{M,M}) = @inbounds (C = similar(A); C[:] = A[:]; for i = 1:n; C = MMatrix{M,M}(SMatrix{M,M}(C) + SMatrix{M,M}(A)); end; return C)

    # Notes: - A[:] = B[:] is allocating in Base, unlike `map!`
    #        - Also, the same goes for Base's implementation of broadcast!(f, A, B, C) (but map! is OK).
    #        - I really need to implement copy() in StaticArrays... (and maybe a special instance of setindex!(C, :))

    # Old, "broken" defs:
    #f(n::Integer, A, B) = @inbounds (C = A*B; for i = 1:n; C += A*B; end; return C)
    #f_mut(n::Integer, A, B) = @inbounds (C = A*B; tmp = similar(C); for i = 1:n; A_mul_B!(tmp, A, B); map!(+, C, C, tmp); end; return C)
    #g(n::Integer, A, B) = @inbounds (C = A + B; for i = 1:n; C = C + B; end; return C)
    #g_mut(n::Integer, A, B) = @inbounds (C = A + B; for i = 1:n; @inbounds broadcast!(+, C, C, B); end; return C)
end

# Test and compile f()

C = f(2, A)
C_mut = f_mut_array(2, A)
if N == 4;  @assert C_mut ≈ C; end

println("=====================================")
println("    Benchmarks for $N×$N matrices")
println("=====================================")
print("SMatrix * SMatrix compilation time (unrolled):         ")
@time eval(quote
    Cs_unrolled = f_unrolled(2, As)
    Cs_unrolled::SMatrix
    if N == 4; @assert Cs_unrolled ≈ C; end
end)

print("SMatrix * SMatrix compilation time (chunks):           ")
@time eval(quote
    Cs_chunks = f_unrolled_chunks(2, As)
    Cs_chunks::SMatrix
    if N == 4; @assert Cs_chunks ≈ C; end
end)

print("MMatrix * MMatrix compilation time (unrolled):         ")
@time eval(quote
    Cm_unrolled = f_unrolled(2, Am)
    Cm_unrolled::MMatrix
    if N == 4; @assert Cm_unrolled ≈ C; end
end)

print("MMatrix * MMatrix compilation time (chunks):           ")
@time eval(quote
    Cm_chunks = f_unrolled_chunks(2, Am)
    Cm_chunks::MMatrix
    if N == 4; @assert Cm_chunks ≈ C; end
end)

print("A_mul_B!(MMatrix, MMatrix) compilation time (unrolled):")
@time eval(quote
    Cm_mut = f_mut_marray(2, Am)
    Cm_mut::MMatrix
    if N == 4; @assert Cm_mut ≈ C; end
end)

print("A_mul_B!(MMatrix, MMatrix) compilation time (BLAS):    ")
@time eval(quote
    Cm_blas = f_blas_marray(2, Am)
    Cm_blas::MMatrix
    if N == 4; @assert Cm_blas ≈ C; end
end)


# Warmup and some checks
Cs = f(2, As)
Cm = f(2, Am)
Cm_via_sarray = f_via_sarray(2, Am)

Cs::SMatrix
if N == 4; @assert Cs ≈ C; end
Cm::MMatrix
if N == 4; @assert Cm ≈ C; end
Cm_via_sarray::MMatrix
if N == 4; @assert Cm_via_sarray ≈ C; end

@static if fsa
    print("Mat * Mat compilation time:                            ")
    @time eval(quote
        C = f(2, A)
        Cf = f(2, Af)
        Cf::Mat
        if N == 4; @assert Cf == C; end
    end)
end

# Test and compile g()

C = g(2, A)
C_mut = g_mut(2, A)

if N == 4; @assert C_mut ≈ C; end

C = g(2, A)
Cs = g(2, As)
Cm = g(2, Am)
Cm_via_sarray = g_via_sarray(2, Am)
Cm_mut = g_mut(2, Am)

Cs::SMatrix
if N == 4; @assert Cs == C; end
Cm::MMatrix
if N == 4; @assert Cm == C; end
Cm_via_sarray::MMatrix
if N == 4; @assert Cm_via_sarray == C; end
Cm_mut::MMatrix
if N == 4; @assert Cm_mut == C; end

@static if fsa
    Cf = g(2, Af)
    Cf::Mat
    if N == 4; @assert Cf == C; end
end

println()

# Do the performance tests

println("Matrix multiplication")
println("---------------------")
begin
   print("Array               ->"); @time f(M_f, A)
   print("Array (mutating)    ->"); @time f_mut_array(M_f, A)
   print("SArray              ->"); @time f(M_f, As)
   print("SArray (unrolled)   ->"); @time f_unrolled(M_f, As)
   print("SArray (chunks)     ->"); @time f_unrolled_chunks(M_f, As)
   print("MArray              ->"); @time f(M_f, Am)
   print("MArray (unrolled)   ->"); @time f_unrolled(M_f, Am)
   print("MArray (chunks)     ->"); @time f_unrolled_chunks(M_f, Am)
   print("MArray (via SArray) ->"); @time f_via_sarray(M_f, Am)
   print("MArray (mutating)   ->"); @time f_mut_marray(M_f, Am)
   print("MArray (BLAS gemm!) ->"); @time f_blas_marray(M_f, Am)
   @static if fsa
       print("Mat                 ->"); @time f(M_f, Af)
   end
end
println()

println("Matrix addition")
println("---------------")
begin
   print("Array               ->"); @time g(M_g, A)
   print("Array (mutating)    ->"); @time g_mut(M_g, A) # broadcast! seems to be broken!
   print("SArray (unrolled)   ->"); @time g(M_g, As)
   print("MArray (unrolled)   ->"); @time g(M_g, Am)
   print("MArray (via SArray) ->"); @time g_via_sarray(M_g, Am)
   print("MArray (mutating)   ->"); @time g_mut(M_g, Am)
   @static if fsa
       print("Mat                 ->"); @time g(M_g, Af)
   end
end
println()
