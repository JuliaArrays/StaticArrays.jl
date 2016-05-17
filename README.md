# StaticArrays

*An experimental playground for statically-sized arrays*

[![Build Status](https://travis-ci.org/andyferris/StaticArrays.jl.svg?branch=master)](https://travis-ci.org/andyferris/StaticArrays.jl)

Currently, this project provides two subclasses of abstract
`StaticArray{Sizes,T,N} <: AbstractArray{T,N}` (where *static* means statically
sized, and `Sizes` is a tuple of `Int`s):

* `SArray` is a stack-allocated immutable array which effectively wraps a
  tuple and aims to replicate the non-mutating part of the `Array` interface.

* `MArray` is a heap-allocated mutable array, which works in parallel to
  `SArray` by wrapping a tuple in the `Array` interface. Being mutable, this
  type provides the most complete implementation as many methods in `Base` work
  correctly, and can be helpful in building frequently-used but infrequently
  constructed `SArray`s.

Similarly, we will have linear algebra on `SVector`s, `SMatrix`s, `MVector`s
and `MMatrix`s, which are simple 1- and 2- dimensional type-aliases.

### Quick exposé

The `SArray` and `MArray` types are defined as

```julia
immutable SArray{Sizes,T,N,D} <: StaticArray{Sizes,T,N}
    data::D # D = NTuple{prod(Sizes), T}
end

type MArray{Sizes,T,N,D} <: StaticArray{Sizes,T,N}
    data::D # D = NTuple{prod(Sizes), T}
end
```

Constructors take a flat tuple (even for higher-dimensional arrays) and should
"just work" and perform similar promotion to `Array`. Also, conversion to `Array` and
from `AbstractArray` types are implemented (though, at minimum, the size of the `StaticArray`
must be provided as a type-parameter). Inner constructors perform
compile-time checking of type-parameters, to make sure `Sizes`, `T`, `N`, and `D`
are consistent.

Currently, arbitrary linear and multi-dimensional scalar indexing are functioning, but
more work is required, especially on the immutable `map` function and operators.
Since `similar` and `setindex!` already work with `MArray`, most functions like
`+` work naturally with `MArray`.

Ultimately, in Julia 0.5, this should be easy to implement for `SArray` where
`.+`, etc, map to `broadcast`. Linear algebra and other complex operations are
yet to be implemented.

### See also

This package takes inspiration from:

[Julep: More support for working with immutables #11902](https://github.com/JuliaLang/julia/issues/11902)

[FixedSizeArrays.jl](https://github.com/SimonDanisch/FixedSizeArrays.jl)

[ImmutableArrays.jl](https://github.com/JuliaGeometry/ImmutableArrays.jl)
