# StaticArrays

*Statically sized arrays for Julia 0.5*

[![StaticArrays](http://pkg.julialang.org/badges/StaticArrays_0.5.svg)](http://pkg.julialang.org/?pkg=StaticArrays)
[![Build Status](https://travis-ci.org/andyferris/StaticArrays.jl.svg?branch=master)](https://travis-ci.org/andyferris/StaticArrays.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/px9kulhngvs26fka?svg=true)](https://ci.appveyor.com/project/andyferris/staticarrays-jl)
[![Coverage Status](https://coveralls.io/repos/github/andyferris/StaticArrays.jl/badge.svg?branch=master)](https://coveralls.io/github/andyferris/StaticArrays.jl?branch=master)

**StaticArrays** provides a framework for implementing statically sized arrays
in Julia (≥ 0.5), using the abstract type `StaticArray{T,N} <: DenseArray{T,N}`.
Subtypes of `StaticArray` will provide fast implementations of common array and
linear algebra operations. Note that here "statically sized" means that the
size can be determined from the *type* (so concrete implementations of
`StaticArray` must define a method `size(::Type{T})`), and "static" does **not**
necessarily imply `immutable`.

The package also provides some concrete static array types: `SVector`, `SMatrix`
and `SArray`, which may be used as-is (or else embedded in your own type).
Mutable versions `MVector`, `MMatrix` and `MArray` are also exported. Further,
the abstract `FieldVector` can be used to make fast `StaticVector`s out of any
uniform Julia "struct".

## Speed

The speed of small `SVector`s, `SMatrix`s and `SArray`s is often > 10 × faster
than `Base.Array`. See this sample benchmark (or see the full results [here](https://github.com/andyferris/StaticArrays.jl/blob/master/perf/bench8.txt)):

```
=====================================
   Benchmarks for 3×3 matrices
=====================================

Matrix multiplication
---------------------
Array  ->  3.973188 seconds (74.07 M allocations: 6.623 GB, 12.92% gc time)
SArray ->  0.326989 seconds (5 allocations: 240 bytes)
MArray ->  2.248258 seconds (37.04 M allocations: 2.759 GB, 14.06% gc time)

Matrix multiplication (mutating)
--------------------------------
Array  ->  2.237091 seconds (6 allocations: 480 bytes)
MArray ->  0.795372 seconds (6 allocations: 320 bytes)

Matrix addition
---------------
Array  ->  2.610709 seconds (44.44 M allocations: 3.974 GB, 11.81% gc time)
SArray ->  0.073024 seconds (5 allocations: 240 bytes)
MArray ->  0.896849 seconds (22.22 M allocations: 1.656 GB, 21.33% gc time)

Matrix addition (mutating)
--------------------------
Array  ->  0.872791 seconds (6 allocations: 480 bytes)
MArray ->  0.145895 seconds (5 allocations: 240 bytes)
```

(Run with `julia -O3` for even faster SIMD code with immutable static arrays!)

## Quick start

```julia
Pkg.add("StaticArrays")  # or Pkg.clone("https://github.com/andyferris/StaticArrays.jl")
using StaticArrays

# Create an SVector using various forms, using constructors, functions or macros
v1 = SVector(1, 2, 3)
v1.data === (1, 2, 3) # SVector uses a tuple for internal storage
v2 = SVector{3,Float64}(1, 2, 3) # length 3, eltype Float64
v3 = @SVector [1, 2, 3]
v4 = @SVector [i^2 for i = 1:10] # arbitrary comprehensions (where range can be evaluated a global scope)
v5 = zeros(SVector{3}) # defaults to Float64
v6 = @SVector zeros(3)

# Can get size() from instance or type
size(v1) == (3,)
size(typeof(v1)) == (3,)

# Similar constructor syntax for matrices
m1 = SMatrix{2,2}(1, 2, 3, 4) # flat, column-major storage, equal to m2:
m2 = @SMatrix [ 1  3 ;
                2  4 ]
m3 = eye(SMatrix{3,3})
m4 = @SMatrix randn(4,4)

# Higher-dimensional support
a = @SArray randn(2, 2, 2, 2, 2, 2)

# Supports all the common operations of AbstractArray
v7 = v1 + v2
v8 = sin.(v3)
v3 == m3 * v3 # m3 = eye(SMatrix{3,3})
# map, reduce, broadcast, map!, broadcast!, etc...

# Indexing also supports tuples
v1[1] === 1
v1[(3,2,1)] === @SVector [3, 2, 1]
v1[:] === v1
typeof(v1[[1,2,3]]) == Vector # Can't determine number of elements from the type of [1,2,3]

# Inherits from DenseArray, so is hooked into BLAS, LAPACK, etc:
rand(MMatrix{20,20}) * rand(MMatrix{20,20}) # large matrices can use BLAS multiplication
eig(m3) # eig(), etc use LAPACK

# Static arrays stay statically sized, even when used by Base functions, etc:
typeof(eig(m3)) == Tuple{StaticArrays.MVector{3,Float64}, StaticArrays.MMatrix{3,3,Float64,9}}

# similar() returns a mutable container, while similar_type() returns a constructor:
typeof(similar(m3)) == MMatrix{3,3,Float64,9} # (final parameter is length = 9)
similar_type(m3) == SMatrix{3,3,Float64,9}
```

## Approach

Primarily, the package provides methods for common `AbstractArray` functions,
specialized for (potentially immutable) statically sized arrays. Many of
Julia's built-in method definitions inherently assume mutability, and further
performance optimizations may be made when the size of the array is know to the
compiler (by loop unrolling, for instance).

At the lowest level, `getindex` on statically sized arrays will call `getfield`
on types or tuples, and in this package `StaticArray`s are limited to
`LinearFast()` access patterns with 1-based indexing. What this means is that
all `StaticArray`s support linear indexing into a dense, column-based storage
format. By simply defining `size(::Type{T})` and `getindex(::T, ::Integer)`,
the `StaticArray` interface will look after multi-dimensional indexing,
`map`/`map!`, `reduce`, `broadcast`/`broadcast!`, matrix multiplication and a
variety of other operations.

Finally, since `StaticArrays <: DenseArray`, many methods such as `sqrtm`,
`eig`, `chol`, and more are already defined in `Base`. Conversion
to pointers let us interact with LAPACK and similar C/Fortran libraries through
the existing `StridedArray` interface. In some instances mutable `StaticArray`s
(`MVector` or `MMatrix`) will be returned, while in other cases the definitions
fall back to `Array`. This approach gives us maximal versatility while retaining
the ability to implement fast specializations in the future.

## Details
### Indexing

Statically sized indexing can be realized by indexing each dimension by a
scalar, an `NTuple{N, Integer}` or `:` (on statically sized arrays only).
Indexing in this way will result a statically sized array (even if the input was
dynamically sized) of the closest type (as defined by `similar_type`).

Conversely, indexing a statically sized array with a dynamically sized index
(such as a `Vector{Integer}` or `UnitRange{Integer}`) will result in a standard
(dynamically sized) `Array`.

### `similar_type()`

Since immutable arrays need to be constructed "all-at-once", we need a way of
obtaining an appropriate constructor if the element type or dimensions of the
output array differs from the input. To this end, `similar_type` is introduced,
behaving just like `similar`, except that it returns a type. Relevant methods
are:

```julia
similar_type{A <: StaticArray}(::Type{A}) # defaults to A
similar_type{A <: StaticArray, ElType}(::Type{A}, ::Type{ElType}) # Change element type
similar_type{A <: StaticArray}(::Type{A}, size::Tuple{Int...}) # Change size
similar_type{A <: StaticArray, ElType}(::Type{A}, ::Type{ElType}, size::Tuple{Int...}) # Change both
```

These setting will affect everything, from indexing, to matrix multiplication
and `broadcast`.

Use of `similar` will fall back to a mutable container, such as a `MVector`
(see below).

### `SVector`

The simplest static array is the `SVector`, defined as

```julia
immutable SVector{N,T} <: StaticVector{T}
    data::NTuple{N,T}
end
```

`SVector` defines a series of convenience constructors, so you can just type
e.g. `SVector(1,2,3)`. Alternatively there is an intelligent `@SVector` macro
where you can use native Julia array literals syntax, comprehensions, and the
and the `zeros()`, `ones()`, `rand()` and `randn()` functions, such as `@SVector [1,2,3]`,
`@SVector Float64[1,2,3]`, `@SVector [f(i) for i = 1:10]`, `@SVector zeros(3)`,
`@SVector randn(Float32, 4)`, etc (Note: the range of a comprehension is evaluated at global scope by the
macro, and must be made of combinations of literal values, functions, or global
variables, but is not limited to just simple ranges. Extending this to
(hopefully statically known by type-inference) local-scope variables is hoped
for the future. The `zeros()`, `ones()`, `rand()` and `randn()` functions do not have this
limitation.)

### `SMatrix`

Static matrices are also provided by `SMatrix`. It's definition is a little
more complicated:

```julia
immutable SMatrix{S1, S2, T, L} <: StaticMatrix{T}
    data::NTuple{L, T}
end
```

Here `L` is the `length` of the matrix, such that `S1 × S2 = L`. However,
convenience constructors are provided, so that `L`, `T` and even `S2` are
unnecessary. At minimum, you can type `SMatrix{2}(1,2,3,4)` to create a 2×2
matrix (the total number of elements must divide evenly into `S1`). A
convenience macro `@SMatrix [1 2; 3 4]` is provided (which also accepts
comprehensions and the `zeros()`, `ones()`, `rand()`, `randn()` and `eye()`
functions).

### `SArray`

A container with arbitrarily many dimensions is defined as
`immutable SArray{Size,T,N,L} <: StaticArray{T,N}`, where
`Size = (S1, S2, ...)` is a tuple of `Int`s. You can easily construct one with
the `@SArray` macro, supporting all the features of `@SVector` and `@SMatrix`
(with higher-dimensional support).

Notably, the main reason `SVector` and `SMatrix` are defined is to make it
easier to define the types without the extra tuple characters (compare
`SVector{3}` to `SArray{(3,)}`). This extra convenience was made possible
because it is so easy to define new `StaticArray` subtypes, and they naturally
work together.

### Mutable arrays: `MVector`, `MMatrix` and `MArray`

These statically sized arrays are identical to the above, but are defined as
mutable Julia `type`s, instead of `immutable`. Because they are mutable, they
allow `setindex!` to be defined (achieved through pointer manipulation, into a
tuple).

As a consequence of Julia's internal implementation, these mutable containers
live on the heap, not the stack. Their memory must be allocated and tracked by
the garbage collector. Nevertheless, there is opportunity for speed
improvements relative to `Base.Array` because (a) there may be one less
pointer indirection, (b) their (typically small) static size allows for
additional loop unrolling and inlining, and consequentially (c) their mutating
methods like `map!` are extremely fast. Benchmarking shows that operations such
as addition and matrix multiplication are faster for `MMatrix` than `Matrix`,
at least for sizes up to 14 × 14, though keep in mind that optimal speed will
be obtained by using mutating functions (like `map!` or `A_mul_B!`) where
possible, rather than reallocating new memory.

Mutable static arrays also happen to be very useful containers that can be
constructed on the heap (with the ability to use `setindex!`, etc), and later
copied as e.g. an immutable `SVector` to the stack for use, or into e.g. an
`Array{SVector}` for storage.

Convenience macros `@MVector`, `@MMatrix` and `@MArray` are provided.

### `FieldVector`

Sometimes it might be useful to imbue your own types, having multiple fields,
with vector-like properties. *StaticArrays* can take care of this for you by
allowing you to inherit from `FieldVector{T}`. For example, consider:

```julia
immutable Point3D <: FieldVector{Float64}
    x::Float64
    y::Float64
    z::Float64
end
```

With this type, users can easily access fields to `p = Point3D(x,y,z)` using
`p.x`, `p.y` or `p.z`, or alternatively via `p[1]`, `p[2]`, or `p[3]`. You may
even permute the coordinates with `p[(3,2,1)]`). Furthermore, `Point3D` is a
complete `AbstractVector` implementation where you can add, subtract or scale
vectors, multiply them by matrices (and return the same type), etc.

It is also worth noting that `FieldVector`s may be mutable or immutable, and
that `setindex!` is defined for use on mutable types. For mutable containers,
you may want to define a default constructor (no inputs) that can be called by
`similar`.

### Implementing your own types

You can easily create your own `StaticArray` type, by defining both `size` on
the type, and linear `getindex` (and optionally `setindex!` for mutable types
- see `setindex(::SVector, val, i)` in *MVector.jl* for an example of how to
achieve this through pointer manipulation). Your type should define a constructor
that takes a tuple of the data (and mutable containers may want to define a
default constructor).

Other useful functions to overload may be `similar_type` (and `similar` for
mutable containers).

### SIMD optimizations

It seems Julia and LLVM are smart enough to use processor vectorization
extensions like SSE and AVX - however they are currently partially disabled by
default. Run Julia with `julia -O` or `julia -O3` to enable these optimizations,
and many of your (immutable) `StaticArray` methods *should* become significantly
faster!

### *FixedSizeArrays* compatibility

You can try `using StaticArrays.FixedSizeArrays` to add some compatibility
wrappers for the most commonly used features of the *FixedSizeArrays* package,
such as `Vec`, `Mat`, `Point` and `@fsa`. These wrappers do not provide a
perfect interface, but may help in trying out *StaticArrays* with pre-existing
code.

### See also

This package takes inspiration from:

* [Julep: More support for working with immutables #11902](https://github.com/JuliaLang/julia/issues/11902)
* [FixedSizeArrays.jl](https://github.com/SimonDanisch/FixedSizeArrays.jl)
* [ImmutableArrays.jl](https://github.com/JuliaGeometry/ImmutableArrays.jl)
