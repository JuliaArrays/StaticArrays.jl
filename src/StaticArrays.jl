module StaticArrays

import Base: @pure, @propagate_inbounds, getindex, setindex!, size,
             length, convert, promote_op, map, reduce, mapreduce,
             broadcast, conj, transpose, ctranspose

export StaticArray, StaticVector, StaticMatrix
export SArray, SVector, SMatrix
export MArray, MVector, MMatrix
export FieldVector

export @SVector, @SMatrix, @SArray

export similar_type

include("util.jl")

include("core.jl")
include("SVector.jl")
include("FieldVector.jl")
include("SMatrix.jl")
#include("SArray.jl")

include("indexing.jl")
include("abstractarray.jl")
include("mapreduce.jl")
include("arraymath.jl")
include("linalg.jl")
include("matrix_multiply.jl")

# TODO list
# ---------
#
#
# * How to deal with functions like reshape, etc - which could work by returning
#   an Array, (reshape could take a Val{}).
#
# * Should similar(::SArray, ...) return a mutable MArray? A Ref{SArray}? Or throw an error? (see also above)
#
# * map, map!, broadcast, broadcast!
#
# * permutedims()
#
# * element-wise ops
#
# * vector and matrix multiplications
#
# * Linear algebra - matrix functions (det, inv, eig, svd, qr, etc...)
#
# * Investigate and potentially improve speed of getindex/setindex!
#
# * @fsa-like macros for easily making matrices. @SArray and @MArray? (or lower case?) (or @static)


end # module
