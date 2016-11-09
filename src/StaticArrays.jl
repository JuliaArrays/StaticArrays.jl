__precompile__()

module StaticArrays

import Base: @pure, @propagate_inbounds, getindex, setindex!, size, similar,
             length, convert, promote_op, map, map!, reduce, mapreduce,
             broadcast, broadcast!, conj, transpose, ctranspose, hcat, vcat,
             ones, zeros, eye, one, cross, vecdot, reshape, fill, fill!, det,
             inv, eig, trace, vecnorm, norm, dot, diagm, sum, prod, count, any,
             all, sumabs, sumabs2, minimum, maximum, extrema, mean

export StaticScalar, StaticArray, StaticVector, StaticMatrix
export Scalar, SArray, SVector, SMatrix
export MArray, MVector, MMatrix
export FieldVector, MutableFieldVector
export SizedArray, SizedVector, SizedMatrix

export Size

export @SVector, @SMatrix, @SArray
export @MVector, @MMatrix, @MArray

export similar_type, setindex

include("util.jl")

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

include("indexing.jl")
include("abstractarray.jl")
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

include("FixedSizeArrays.jl")
include("ImmutableArrays.jl")

# TODO list
# ---------
#
# * more tests
#
# * reshape() - accept Val? Currently uses `ReshapedArray`. Cool :)
#
# * permutedims() - accept Val? Or wait for `PermutedDimsArray` ?
#
# * Linear algebra - matrix functions (det, inv, eig, svd, qr, etc...)
#                    (currently, we use pointers to interact with LAPACK, etc)


end # module
