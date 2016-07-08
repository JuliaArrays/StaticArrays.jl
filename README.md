# StaticArrays

*Statically-sized arrays for Julia 0.5*

[![Build Status](https://travis-ci.org/andyferris/StaticArrays.jl.svg?branch=master)](https://travis-ci.org/andyferris/StaticArrays.jl)

**StaticArrays** provides a framework for implementing statically sized arrays
in Julia (≥ 0.5), using the abstract type `StaticArray{T,N} <: AbstractArray{T,N}`.
Subtypes of `StaticArray` will provide fast implementations of common array and
linear algebra operations. Note that here "statically sized" means that the
size can be determined from the *type* (so concrete implementations of
`StaticArray` must define a method `size(::Type{T})`), and "static" does *not*
necessarily imply `immutable`.

The package also provides some concrete static array types: `SVector`, `SMatrix`
and `SArray`, which may be used as-is (or else embedded in your own type).
Mutable versions `MVector`, `MMatrix` and `MArray` are also exported.
Further, the abstract `FieldVector` can be used to make fast `StaticVector`s out
of any uniform Julia "struct".

### Approach

Primarily, the package provides methods for common `AbstractVector` functions,
specialized for (possibly immutable) statically sized arrays. Many of Julia's
built-in method definitions inherently assume mutability, and further
performance optimizations may be made when the size of the array is know to the
compiler (by loop unrolling, for instance).

At the lowest level, `getindex` on statically sized arrays will call `getfield` on
types or tuples, and in this package `StaticArray`s are limited to `LinearFast()` access patterns with 1-based indexing.
What this means is that all `StaticArray`s support linear indexing into a
dense, column-based storage format. By simply defining `size(::Type{T})` and
`getindex(::T, ::Integer)`, the `StaticArray` interface will look after
multi-dimensional indexing, `map`, `reduce`, `broadcast`, matrix multiplication and
a variety of other operations.

### Indexing

Statically sized indexing can be realized by indexing each dimension by a
scalar, a `NTuple{N, Integer}` or `:` (on statically sized arrays only).
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

Use of `similar` will fall back to a mutable container, such as a `MVector` (see below).

### `SVector`

The simplest static array is the `SVector`, defined as

```julia
immutable SVector{N,T} <: StaticVector{T}
    data::NTuple{N,T}
end
```

`SVector` defines a series of convenience constructors and the `@SVector` macro,
so you can just type `SVector(1,2,3)` or `@SVector [1,2,3]`.

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
convenience macro `@SMatrix [1 2; 3 4]` is provided.

### `SArray`

A container with arbitrarily many dimensions is defined as
`immutable SArray{Size,T,N,L} <: StaticArray{T,N}`, where
`Size = (S1, S2, ...)` is a tuple of `Int`s.

Notably, the main reason `SVector` and `SMatrix` are defined is to make it
easier to define the types without the extra tuple characters (compare
`SVector{3}` to `SArray{(3,)}`). This extra convenience was made possible
because it is so easy to define new `StaticArray` subtypes.

### Mutable arrays: `MVector`, `MMatrix` and `MArray`

These statically-sized arrays are identical to the above, but are defined as
mutable Julia `type`s, instead of `immutable`. Because they are mutable, they
allow `setindex!` to be defined (achieved through pointer manipulation).

As a consequence of Julia's internal implementation, these mutable containers
live on the heap, not the stack. Their memory must be allocated and tracked by
the garbage collector. Nevertheless, there is opportunity for speed
improvements relative to `Base.Array` because (a) there may be one less
pointer indirection and (b) their (typically small) static size allows for
additional loop unrolling and inlining. They are also very useful containers
that can be constructed on the heap and later copied as e.g. an immutable
`SVector` to the stack for use, or into e.g. an `Array{SVector}` for storage.

Convenience macros `@MVector` and `@MMatrix` are provided.

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
vectors, multiply them by matrices, etc.

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
mutable containers)

### SIMD optimizations

It seems Julia and LLVM are smart enough to use processor vectorization extensions
like SSE and AVX - however they are disabled by default. Run Julia with
`julia -O` or `julia -O3` to enable these optimizations, and many of your
`StaticArray` methods *may* become significantly faster!

### See also

This package takes inspiration from:

* [Julep: More support for working with immutables #11902](https://github.com/JuliaLang/julia/issues/11902)
* [FixedSizeArrays.jl](https://github.com/SimonDanisch/FixedSizeArrays.jl)
* [ImmutableArrays.jl](https://github.com/JuliaGeometry/ImmutableArrays.jl)
