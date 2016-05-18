module StaticArrays

export StaticArray, StaticVector, StaticMatrix
export SArray, SVector, SMatrix
export MArray, MVector, MMatrix

export similar_type

include("util.jl")
include("core.jl")
include("indexing.jl")
include("abstractarray.jl")
include("map.jl")
include("linalg.jl")

# TODO list
# ---------
#
# * Decide if indexing with collections of indeterminate size should be an error
#   or result in a dynamically-sized Array
#
# * Similarly for other functions - reshape, etc - which could work by returning
#   an Array, but currently results in an error (now, reshape takes a Val{}).
#
# * Implement both getindex/unsafe_getindex (setindex! and unsafe_setindex! already done)
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
#

end # module
