module StaticArrays

export StaticArray, StaticVector, StaticMatrix
export SArray, SVector, SMatrix
export MArray, MVector, MMatrix

export similar_type

include("util.jl")
include("core.jl")
include("abstractarray.jl")
include("map.jl")
include("linalg.jl")

end # module
