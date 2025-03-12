# Static Arrays
*Statically sized arrays for Julia*

**StaticArrays** provides a framework for implementing statically sized arrays
in Julia, using the abstract type `StaticArray{Size,T,N} <: AbstractArray{T,N}`.
Subtypes of [`StaticArray`](@ref) will provide fast implementations of common array and
linear algebra operations. Note that here "statically sized" means that the
size can be determined from the *type*, and "static" does **not** necessarily
imply `immutable`.

The package also provides some concrete static array types: [`SVector`](@ref), [`SMatrix`](@ref)
and [`SArray`](@ref), which may be used as-is (or else embedded in your own type).
Mutable versions [`MVector`](@ref), [`MMatrix`](@ref) and [`MArray`](@ref) are also exported, as well
as [`SizedArray`](@ref) for annotating standard `Array`s with static size information.
Further, the abstract [`FieldVector`](@ref) can be used to make fast static vectors
out of any uniform Julia "struct".

## When Static Arrays may be useful

When a program uses many small ($\lesssim 100$ elements) fixed-sized arrays (whose size is known by the compiler, i.e. "statically," when the performance-critical code is compiled), then using Static Arrays can have several performance advantages:

1. Completely unrolling to loop-free code.  e.g. `v3 = v1 + v2` is just implemented as 3 scalar additions, with no loop overhead at all, if these are all `SVector{3, Float64}`.  The unrolled loops can also sometimes trigger SIMD optimizations.
2. Avoiding heap (or even stack) allocations of temporary arrays.  This is related to point (1) — working with static arrays as local variables is *as if* you had just written out all of the scalar operations by hand on all the components.
3. Being stored "inline" in other data structures.  e.g. a length-$N$ array `Vector{SVector{3, Float64}}`, such as `[v1, v2, v3]` from the `SVector`s in the previous example, is stored as $3N$ *consecutive* `Float64` values.  This is *much* more efficient to access than a `Vector{Vector{Float64}}`, which is essentially an array of pointers to `Vector{Float64}` arrays, and is often faster and more convenient than a $3 \times N$ `Matrix{Float64}` (e.g. because the length 3 is stored in the type you get the benefits of (1) and (2) when accessing vectors in the array, and no "slicing" or "views" are needed).  (There is also [HybridArrays.jl](https://github.com/JuliaArrays/HybridArrays.jl) for matrices with a static number of rows.)

## When Static Arrays may be less useful

You probably don't want to use Static Arrays if:

1. The size of the array is not static.  For example, if it is changing too rapidly at runtime to make it worthwhile to recompile (and/or dynamically dispatch) the code every time the size changes.  (Such as when you have a collection of vectors of many different lengths.)
2. The size of the array is large ($\gg 100$ elements) You (1) don't want to unroll vector operations of long lengths because the code size explodes; (2) they need to be heap-allocated because you can't store them in registers or perhaps even on the stack; and (3) you can already get many of the benefits of "inline" storage in arrays by storing them as the columns of a `Matrix`, whereas you wouldn't want to store a large array inline in an arbitrary `struct`.
3. Working with the array is not performance critical — static arrays are a little less convenient to work with than an ordinary `Array` (because the size of the former is a constant encoded in the type), and it may not be worth the inconvenience for cases where performance is not paramount.
