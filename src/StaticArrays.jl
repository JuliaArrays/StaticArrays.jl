__precompile__()

module StaticArrays

import Base: @_inline_meta, @_propagate_inbounds_meta, @_pure_meta, @propagate_inbounds, @pure

import Base: getindex, setindex!, size, similar,
             length, convert, promote_op, map, map!, reduce, reducedim,
             mapreduce, broadcast, broadcast!, conj, transpose, ctranspose,
             hcat, vcat, ones, zeros, eye, one, cross, vecdot, reshape, fill,
             fill!, det, inv, eig, eigvals, trace, vecnorm, norm, dot, diagm,
             sum, diff, prod, count, any, all, sumabs, sumabs2, minimum,
             maximum, extrema, mean, copy

using Compat

export StaticScalar, StaticArray, StaticVector, StaticMatrix
export Scalar, SArray, SVector, SMatrix
export MArray, MVector, MMatrix
export FieldVector, MutableFieldVector
export SizedArray, SizedVector, SizedMatrix

export Size, Length

export @SVector, @SMatrix, @SArray
export @MVector, @MMatrix, @MArray

export similar_type
export push, pop, shift, unshift, insert, deleteat, setindex

include("util.jl")
include("generated.jl")

include("core.jl")
include("traits.jl")
include("Scalar.jl")
include("SVector.jl")
include("FieldVector.jl")
include("SMatrix.jl")
include("SArray.jl")
include("MVector.jl")
include("MMatrix.jl")
include("MArray.jl")
include("SizedArray.jl")


include("abstractarray.jl")
include("indexing.jl") #=
include("mapreduce.jl")
include("arraymath.jl")
include("linalg.jl")
include("matrix_multiply.jl")
include("solve.jl")
include("deque.jl")
include("det.jl")
include("inv.jl")
include("eigen.jl")
include("cholesky.jl")
=#


if VERSION < v"0.6.0-dev.1671"
    include("FixedSizeArrays.jl")
end
include("ImmutableArrays.jl")


end # module
