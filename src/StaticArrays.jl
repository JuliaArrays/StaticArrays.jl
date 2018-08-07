module StaticArrays

import Base: @_inline_meta, @_propagate_inbounds_meta, @_pure_meta, @propagate_inbounds, @pure

import Base: getindex, setindex!, size, similar, vec, show, length, convert, promote_op,
             promote_rule, map, map!, reduce, mapreduce, broadcast,
             broadcast!, conj, hcat, vcat, ones, zeros, one, reshape, fill, fill!, inv,
             iszero, sum, prod, count, any, all, minimum, maximum, extrema,
             copy, read, read!, write

if VERSION >= v"0.7.0-beta.85"
    import Statistics: mean
else
    import Base: mean
end

using Random
import Random: rand, randn, randexp, rand!, randn!, randexp!
using Core.Compiler: return_type
import Base: sqrt, exp, log
using LinearAlgebra
import LinearAlgebra: transpose, adjoint, eigvals, eigen, lyap, tr,
                      kron, diag, norm, dot, diagm, lu, svd, svdvals,
                      factorize, ishermitian, issymmetric, isposdef, normalize,
                      normalize!, Eigen, det, logdet, cross, diff, qr
# import eye for deprecation warnings
@static if isdefined(LinearAlgebra, :eye)
    import LinearAlgebra: eye
end

export StaticScalar, StaticArray, StaticVector, StaticMatrix
export Scalar, SArray, SVector, SMatrix
export MArray, MVector, MMatrix
export FieldVector
export SizedArray, SizedVector, SizedMatrix
export SDiagonal

export Size, Length

export @SVector, @SMatrix, @SArray
export @MVector, @MMatrix, @MArray

export similar_type
export push, pop, pushfirst, popfirst, insert, deleteat, setindex

"""
    abstract type StaticArray{S, T, N} <: AbstractArray{T, N} end
    StaticScalar{T}     = StaticArray{Tuple{}, T, 0}
    StaticVector{N,T}   = StaticArray{Tuple{N}, T, 1}
    StaticMatrix{N,M,T} = StaticArray{Tuple{N,M}, T, 2}

`StaticArray`s are Julia arrays with fixed, known size.

## Dev docs

They must define the following methods:
 - Constructors that accept a flat tuple of data.
 - `getindex()` with an integer (linear indexing) (preferably `@inline` with `@boundscheck`).
 - `Tuple()`, returning the data in a flat Tuple.

It may be useful to implement:

- `similar_type(::Type{MyStaticArray}, ::Type{NewElType}, ::Size{NewSize})`, returning a
  type (or type constructor) that accepts a flat tuple of data.

For mutable containers you may also need to define the following:

 - `setindex!` for a single element (linear indexing).
 - `similar(::Type{MyStaticArray}, ::Type{NewElType}, ::Size{NewSize})`.
 - In some cases, a zero-parameter constructor, `MyStaticArray{...}()` for unintialized data
   is assumed to exist.

(see also `SVector`, `SMatrix`, `SArray`, `MVector`, `MMatrix`, `MArray`, `SizedArray` and `FieldVector`)
"""
abstract type StaticArray{S <: Tuple, T, N} <: AbstractArray{T, N} end
const StaticScalar{T} = StaticArray{Tuple{}, T, 0}
const StaticVector{N, T} = StaticArray{Tuple{N}, T, 1}
const StaticMatrix{N, M, T} = StaticArray{Tuple{N, M}, T, 2}
const StaticVecOrMat{T} = Union{StaticVector{<:Any, T}, StaticMatrix{<:Any, <:Any, T}}

const AbstractScalar{T} = AbstractArray{T, 0} # not exported, but useful none-the-less
const StaticArrayNoEltype{S, N, T} = StaticArray{S, T, N}

include("util.jl")
include("traits.jl")
include("convert.jl")

include("SUnitRange.jl")
include("FieldVector.jl")
include("SArray.jl")
include("SMatrix.jl")
include("SVector.jl")
include("Scalar.jl")
include("MArray.jl")
include("MVector.jl")
include("MMatrix.jl")
include("SizedArray.jl")
include("SDiagonal.jl")

include("abstractarray.jl")
include("indexing.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("arraymath.jl")
include("linalg.jl")
include("matrix_multiply.jl")
include("det.jl")
include("inv.jl")
include("solve.jl")
include("eigen.jl")
include("expm.jl")
include("sqrtm.jl")
include("lyap.jl")
include("triangular.jl")
include("cholesky.jl")
include("svd.jl")
include("lu.jl")
include("qr.jl")
include("deque.jl")
include("io.jl")

include("FixedSizeArrays.jl")
include("ImmutableArrays.jl")

end # module
