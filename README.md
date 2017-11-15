# StaticArrays

*Statically sized arrays for Julia*

[![StaticArrays](http://pkg.julialang.org/badges/StaticArrays_0.5.svg)](http://pkg.julialang.org/?pkg=StaticArrays)
[![StaticArrays](http://pkg.julialang.org/badges/StaticArrays_0.6.svg)](http://pkg.julialang.org/detail/StaticArrays)
[![Build Status](https://travis-ci.org/JuliaArrays/StaticArrays.jl.svg?branch=master)](https://travis-ci.org/JuliaArrays/StaticArrays.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/xabgh1yhsjxlp30d?svg=true)](https://ci.appveyor.com/project/JuliaArrays/staticarrays-jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaArrays/StaticArrays.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaArrays/StaticArrays.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaArrays/StaticArrays.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaArrays/StaticArrays.jl?branch=master)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaArrays.github.io/StaticArrays.jl/latest)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaArrays.github.io/StaticArrays.jl/stable)

**StaticArrays** provides a framework for implementing statically sized arrays
in Julia (≥ 0.5), using the abstract type `StaticArray{Size,T,N} <: AbstractArray{T,N}`.
Subtypes of `StaticArray` will provide fast implementations of common array and
linear algebra operations. Note that here "statically sized" means that the
size can be determined from the *type*, and "static" does **not** necessarily
imply `immutable`.

The package also provides some concrete static array types: `SVector`, `SMatrix`
and `SArray`, which may be used as-is (or else embedded in your own type).
Mutable versions `MVector`, `MMatrix` and `MArray` are also exported, as well
as `SizedArray` for annotating standard `Array`s with static size information.
Further, the abstract `FieldVector` can be used to make fast `StaticVector`s
out of any uniform Julia "struct".
Full documentation can be found [here](https://JuliaArrays.github.io/StaticArrays.jl/stable/).

## Speed

The speed of *small* `SVector`s, `SMatrix`s and `SArray`s is often > 10 × faster
than `Base.Array`. See this simplified benchmark (or see the full results [here](https://github.com/andyferris/StaticArrays.jl/blob/master/perf/bench10.txt)):

```
============================================
    Benchmarks for 3×3 Float64 matrices
============================================

Matrix multiplication               -> 8.2x speedup
Matrix multiplication (mutating)    -> 3.1x speedup
Matrix addition                     -> 45x speedup
Matrix addition (mutating)          -> 5.1x speedup
Matrix determinant                  -> 170x speedup
Matrix inverse                      -> 125x speedup
Matrix symmetric eigendecomposition -> 82x speedup
Matrix Cholesky decomposition       -> 23.6x speedup
```

These results improve significantly when using `julia -O3` with immutable static
arrays, as the extra optimization results in surprisingly good SIMD code.

Note that in the current implementation, working with large `StaticArray`s puts a
lot of stress on the compiler, and becomes slower than `Base.Array` as the size
increases.  A very rough rule of thumb is that you should consider using a
normal `Array` for arrays larger than 100 elements. For example, the performance
crossover point for a matrix multiply microbenchmark seems to be about 11x11 in
julia 0.5 with default optimizations.


## Quick start

```julia
Pkg.add("StaticArrays")  # or Pkg.clone("https://github.com/JuliaArrays/StaticArrays.jl")
using StaticArrays

# Create an SVector using various forms, using constructors, functions or macros
v1 = SVector(1, 2, 3)
v1.data === (1, 2, 3) # SVector uses a tuple for internal storage
v2 = SVector{3,Float64}(1, 2, 3) # length 3, eltype Float64
v3 = @SVector [1, 2, 3]
v4 = @SVector [i^2 for i = 1:10] # arbitrary comprehensions (range is evaluated at global scope)
v5 = zeros(SVector{3}) # defaults to Float64
v6 = @SVector zeros(3)
v7 = SVector{3}([1, 2, 3]) # Array conversions must specify size

# Can get size() from instance or type
size(v1) == (3,)
size(typeof(v1)) == (3,)

# Similar constructor syntax for matrices
m1 = SMatrix{2,2}(1, 2, 3, 4) # flat, column-major storage, equal to m2:
m2 = @SMatrix [ 1  3 ;
                2  4 ]
m3 = eye(SMatrix{3,3})
m4 = @SMatrix randn(4,4)
m5 = SMatrix{2,2}([1 3 ; 2 4]) # Array conversions must specify size

# Higher-dimensional support
a = @SArray randn(2, 2, 2, 2, 2, 2)

# Supports all the common operations of AbstractArray
v7 = v1 + v2
v8 = sin.(v3)
v3 == m3 * v3 # recall that m3 = eye(SMatrix{3,3})
# map, reduce, broadcast, map!, broadcast!, etc...

# Indexing can also be done using static arrays of integers
v1[1] === 1
v1[SVector(3,2,1)] === @SVector [3, 2, 1]
v1[:] === v1
typeof(v1[[1,2,3]]) <: Vector # Can't determine size from the type of [1,2,3]

# Is (partially) hooked into BLAS, LAPACK, etc:
rand(MMatrix{20,20}) * rand(MMatrix{20,20}) # large matrices can use BLAS
eig(m3) # eig(), etc uses specialized algorithms up to 3×3, or else LAPACK

# Static arrays stay statically sized, even when used by Base functions, etc:
typeof(eig(m3)) == Tuple{SVector{3,Float64}, SMatrix{3,3,Float64,9}}

# similar() returns a mutable container, while similar_type() returns a constructor:
typeof(similar(m3)) == MMatrix{3,3,Float64,9} # (final parameter is length = 9)
similar_type(m3) == SMatrix{3,3,Float64,9}

# The Size trait is a compile-time constant representing the size
Size(m3) === Size(3,3)

# A standard Array can be wrapped into a SizedArray
m4 = Size(3,3)(rand(3,3))
inv(m4) # Take advantage of specialized fast methods

# reshape() uses Size() or types to specify size:
reshape([1,2,3,4], Size(2,2)) == @SMatrix [ 1  3 ;
                                            2  4 ]
typeof(reshape([1,2,3,4], Size(2,2))) === SizedArray{(2, 2),Int64,2,1}

```

## Approach

The package provides a range of different useful built-in `StaticArray` types,
which include mutable and immutable arrays based upon tuples, arrays based upon
structs, and wrappers of `Array`. There is a relatively simple interface for
creating your own, custom `StaticArray` types, too.

This package also provides methods for a wide range of `AbstractArray` functions,
specialized for (potentially immutable) `StaticArray`s. Many of Julia's
built-in method definitions inherently assume mutability, and further
performance optimizations may be made when the size of the array is known to the
compiler. One example of this is by loop unrolling, which has a substantial
effect on small arrays and tends to automatically trigger LLVM's SIMD
optimizations. Another way performance is boosted is by providing specialized
methods for `det`, `inv`, `eig` and `chol` where the algorithm depends on the
precise dimensions of the input. In combination with intelligent fallbacks to
the methods in Base, we seek to provide a comprehensive support for statically
sized arrays, large or small, that hopefully "just works".

## Relationship to *FixedSizeArrays* and *ImmutableArrays*

Several existing packages for statically sized arrays have been developed for
Julia, noteably *FixedSizeArrays* and *ImmutableArrays* which provided signficant
inspiration for this package. Upon consultation, it has been decided to move
forward with *StaticArrays* which has found a new home in the *JuliaArrays*
github organization. It is recommended that new users use this package, and
that existing dependent packages consider switching to *StaticArrays* sometime
during the life-cycle of Julia v0.5.

You can try `using StaticArrays.FixedSizeArrays` to add some compatibility
wrappers for the most commonly used features of the *FixedSizeArrays* package,
such as `Vec`, `Mat`, `Point` and `@fsa`. These wrappers do not provide a
perfect interface, but may help in trying out *StaticArrays* with pre-existing
code.

Furthermore, `using StaticArrays.ImmutableArrays` will let you use the typenames
from the *ImmutableArrays* package, which does not include the array size as a
type parameter (e.g. `Vector3{T}` and `Matrix3x3{T}`).
