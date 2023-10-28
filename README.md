# StaticArrays

*Statically sized arrays for Julia*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaArrays.github.io/StaticArrays.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaArrays.github.io/StaticArrays.jl/dev)
[![Build Status](https://github.com/JuliaArrays/StaticArrays.jl/workflows/CI/badge.svg)](https://github.com/JuliaArrays/StaticArrays.jl/actions?query=workflow%3ACI)
[![codecov.io](https://codecov.io/github/JuliaArrays/StaticArrays.jl/branch/master/graph/badge.svg)](http://codecov.io/github/JuliaArrays/StaticArrays.jl/branch/master)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![deps](https://juliahub.com/docs/StaticArrays/deps.svg)](https://juliahub.com/ui/Packages/StaticArrays/yY9vm?t=2)
[![version](https://juliahub.com/docs/StaticArrays/version.svg)](https://juliahub.com/ui/Packages/StaticArrays/yY9vm)


**StaticArrays** provides a framework for implementing statically sized arrays
in Julia, using the abstract type `StaticArray{Size,T,N} <: AbstractArray{T,N}`.
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

Most of the primary array types exported by StaticArrays.jl are defined in the small interface
package [StaticArraysCore.jl](https://github.com/JuliaArrays/StaticArraysCore.jl). This includes
e.g., the definitions of the abstract type `StaticArray` and the concrete types `SArray`,
`MArray`, and `SizedArray` (as well as their dimension-specific aliases).
This enables downstream packages to implement new methods for these types without depending
on (and hence loading) the entirety of StaticArrays.jl, and thereby to avoid incurring the full
load-time of StaticArrays.jl (which is on the order of 0.6 s for StaticArrays.jl v1.4 on Julia
v1.7).

## Speed

The speed of *small* `SVector`s, `SMatrix`s and `SArray`s is often > 10 × faster
than `Base.Array`. For example, here's a
[microbenchmark](perf/README_benchmarks.jl) showing some common operations.

```
============================================
    Benchmarks for 3×3 Float64 matrices
============================================
Matrix multiplication               -> 5.9x speedup
Matrix multiplication (mutating)    -> 1.8x speedup
Matrix addition                     -> 33.1x speedup
Matrix addition (mutating)          -> 2.5x speedup
Matrix determinant                  -> 112.9x speedup
Matrix inverse                      -> 67.8x speedup
Matrix symmetric eigendecomposition -> 25.0x speedup
Matrix Cholesky decomposition       -> 8.8x speedup
Matrix LU decomposition             -> 6.1x speedup
Matrix QR decomposition             -> 65.0x speedup
```

These numbers were generated on an Intel i7-7700HQ using Julia-1.2. As with all
synthetic benchmarks, the speedups you see here should only be taken as very
roughly indicative of the speedup you may see in real code. When in doubt,
benchmark your real application!

Note that in the current implementation, working with large `StaticArray`s puts a
lot of stress on the compiler, and becomes slower than `Base.Array` as the size
increases.  A very rough rule of thumb is that you should consider using a
normal `Array` for arrays larger than 100 elements.


## Quick start

Add *StaticArrays* from the [Pkg REPL](https://docs.julialang.org/en/latest/stdlib/Pkg/#Getting-Started-1), i.e., `pkg> add StaticArrays`. Then:
```julia
using LinearAlgebra
using StaticArrays

# Use the convenience constructor type `SA` to create vectors and matrices
SA[1, 2, 3]      isa SVector{3,Int}
SA_F64[1, 2, 3]  isa SVector{3,Float64}
SA_F32[1, 2, 3]  isa SVector{3,Float32}
SA[1 2; 3 4]     isa SMatrix{2,2,Int}
SA_F64[1 2; 3 4] isa SMatrix{2,2,Float64}

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
m3 = SMatrix{3,3}(1I)
m4 = @SMatrix randn(4,4)
m5 = SMatrix{2,2}([1 3 ; 2 4]) # Array conversions must specify size

# Higher-dimensional support
a = @SArray randn(2, 2, 2, 2, 2, 2)

# Supports all the common operations of AbstractArray
v7 = v1 + v2
v8 = sin.(v3)
v3 == m3 * v3 # recall that m3 = SMatrix{3,3}(1I)
# map, reduce, broadcast, map!, broadcast!, etc...

# Indexing can also be done using static arrays of integers
v1[1] === 1
v1[SVector(3,2,1)] === @SVector [3, 2, 1]
v1[:] === v1
typeof(v1[[1,2,3]]) <: Vector # Can't determine size from the type of [1,2,3]

# Is (partially) hooked into BLAS, LAPACK, etc:
rand(MMatrix{20,20}) * rand(MMatrix{20,20}) # large matrices can use BLAS
eigen(m3) # eigen(), etc uses specialized algorithms up to 3×3, or else LAPACK

# Static arrays stay statically sized, even when used by Base functions, etc:
typeof(eigen(m3).vectors) == SMatrix{3,3,Float64,9}
typeof(eigen(m3).values) == SVector{3,Float64}

# similar() returns a mutable container, while similar_type() returns a constructor:
typeof(similar(m3)) == MArray{Tuple{3,3},Int64,2,9} # (final parameter is length = 9)
similar_type(m3) == SArray{Tuple{3,3},Int64,2,9}

# The Size trait is a compile-time constant representing the size
Size(m3) === Size(3,3)

# A standard Array can be wrapped into a SizedArray
m4 = SizedMatrix{3,3}(rand(3,3))
inv(m4) # Take advantage of specialized fast methods

# reshape() uses Size() or types to specify size:
reshape([1,2,3,4], Size(2,2)) == @SMatrix [ 1  3 ;
                                            2  4 ]
typeof(reshape([1,2,3,4], Size(2,2))) === SizedArray{Tuple{2, 2},Int64,2,1}

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
methods for `det`, `inv`, `eigen` and `cholesky` where the algorithm depends on the
precise dimensions of the input. In combination with intelligent fallbacks to
the methods in Base, we seek to provide a comprehensive support for statically
sized arrays, large or small, that hopefully "just works".
